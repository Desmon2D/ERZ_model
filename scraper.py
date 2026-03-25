"""
=============================================================================
Image Scraper — Скачивание датасета по URL из JSON
=============================================================================
Rate limit: 3 запроса в секунду.
Поддерживает:
  - resume (пропускает уже скачанные)
  - blacklist.json (пропускает мусорные файлы из предыдущей фильтрации)
  - лимит фото на категорию

Использование:
    # Повторный скрапинг — пропустить мусор из старого датасета
    python scraper.py --input Dataset_prod.json datatest.json --output ./dataset --blacklist blacklist.json --max_per_category 500
=============================================================================
"""

import os
import json
import time
import hashlib
import argparse
import logging
from collections import Counter, defaultdict
from urllib.parse import urlparse, parse_qs
import random

import requests
from PIL import Image
from io import BytesIO


REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
RETRY_DELAY = 2


def setup_logging(output_dir):
    log_path = os.path.join(output_dir, "scraper.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, max_per_second=3):
        self.min_interval = 1.0 / max_per_second
        self.last_request_time = 0.0

    def wait(self):
        now = time.monotonic()
        elapsed = now - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.monotonic()


def download_image(url, session, rate_limiter):
    for attempt in range(MAX_RETRIES):
        try:
            rate_limiter.wait()
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                return response.content
            elif response.status_code == 404:
                return None
            elif response.status_code == 429:
                time.sleep(RETRY_DELAY * (attempt + 2))
            else:
                logging.warning(f"HTTP {response.status_code} для {url}")
        except requests.exceptions.Timeout:
            logging.warning(f"Timeout (попытка {attempt+1}/{MAX_RETRIES})")
        except requests.exceptions.ConnectionError:
            logging.warning(f"Connection error (попытка {attempt+1}/{MAX_RETRIES})")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            logging.error(f"Ошибка: {e}")
            return None
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)
    return None


def get_filename_from_url(url):
    params = parse_qs(urlparse(url).query)
    if "filename" in params:
        return params["filename"][0]
    return hashlib.md5(url.encode()).hexdigest()


def scrape(input_files, output_dir, rate_limit, max_per_category, blacklist_path):
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)

    # Загрузка URL
    all_items = []
    for fpath in input_files:
        with open(fpath, "r", encoding="utf-8") as f:
            items = json.load(f)
        logger.info(f"Загружено {len(items)} записей из {fpath}")
        all_items.extend(items)

    # Дедупликация
    seen = set()
    unique = []
    for item in all_items:
        url = item["concat"]
        if url not in seen:
            seen.add(url)
            unique.append(item)
    logger.info(f"Уникальных URL: {len(unique)} (дубликатов: {len(all_items)-len(unique)})")

    # Загружаем blacklist — пропускаем мусорные файлы
    bad_files = set()
    if blacklist_path and os.path.exists(blacklist_path):
        with open(blacklist_path, "r") as f:
            bl = json.load(f)
        bad_files = set(bl.get("blacklist", []))
        logger.info(f"Blacklist загружен: {len(bad_files)} мусорных (будут пропущены)")

    # Фильтруем мусорные URL до скачивания
    if bad_files:
        before = len(unique)
        unique = [item for item in unique
                  if get_filename_from_url(item["concat"]) not in bad_files]
        logger.info(f"Отфильтровано по blacklist: {before - len(unique)} мусорных URL")

    # Лимит на категорию
    if max_per_category > 0:
        random.seed(42)
        by_cat = defaultdict(list)
        for item in unique:
            by_cat[item["id"]].append(item)
        limited = []
        for cat_id, items in by_cat.items():
            if len(items) > max_per_category:
                random.shuffle(items)
                items = items[:max_per_category]
            limited.extend(items)
        skipped_by_limit = len(unique) - len(limited)
        unique = limited
        logger.info(f"Лимит {max_per_category}/категорию: {len(unique)} (пропущено {skipped_by_limit})")

    cat_counts = Counter(item["id"] for item in unique)
    for cat_id in cat_counts:
        os.makedirs(os.path.join(output_dir, str(cat_id), "d"), exist_ok=True)

    # Уже скачанные — пропускаем
    already = set()
    for cat_id in cat_counts:
        cat_dir = os.path.join(output_dir, str(cat_id), "d")
        if os.path.exists(cat_dir):
            for fname in os.listdir(cat_dir):
                already.add(os.path.splitext(fname)[0])
    logger.info(f"Уже скачано: {len(already)} (будут пропущены)")

    # Скрапинг
    rate_limiter = RateLimiter(max_per_second=rate_limit)
    session = requests.Session()
    session.headers.update({"User-Agent": "ViolationClassifier/1.0"})

    downloaded = 0
    skipped = 0
    failed = 0
    total = len(unique)
    start_time = time.time()
    unique.sort(key=lambda x: x["id"])

    for i, item in enumerate(unique):
        cat_id = item["id"]
        url = item["concat"]
        filename = get_filename_from_url(url)

        if filename in already:
            skipped += 1
            continue

        data = download_image(url, session, rate_limiter)
        if data is None:
            failed += 1
            logger.warning(f"[FAIL] cat={cat_id} url={url}")
            continue

        try:
            img = Image.open(BytesIO(data))
            fmt = (img.format or "JPEG").lower()
            ext = {"jpeg": ".jpg", "png": ".png", "webp": ".webp"}.get(fmt, ".jpg")
        except Exception:
            ext = ".jpg"

        save_path = os.path.join(output_dir, str(cat_id), "d", f"{filename}{ext}")
        with open(save_path, "wb") as f:
            f.write(data)

        downloaded += 1
        already.add(filename)

        processed = downloaded + skipped + failed
        if processed % 50 == 0 or processed == total:
            elapsed = time.time() - start_time
            speed = downloaded / elapsed if elapsed > 0 else 0
            eta = (total - processed) / max(processed / elapsed, 0.001) / 60 if elapsed > 0 else 0
            logger.info(
                f"[{processed}/{total}] "
                f"OK={downloaded} skip={skipped} fail={failed} | "
                f"{speed:.1f} img/s | ETA: {eta:.0f} мин"
            )

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("ГОТОВО")
    logger.info(f"  Скачано:   {downloaded}")
    logger.info(f"  Пропущено: {skipped}")
    logger.info(f"  Ошибки:    {failed}")
    logger.info(f"  Время:     {elapsed/60:.1f} мин")

    logger.info("\nФото по категориям:")
    for cat_id in sorted(cat_counts.keys()):
        cat_dir = os.path.join(output_dir, str(cat_id), "d")
        actual = len(os.listdir(cat_dir)) if os.path.exists(cat_dir) else 0
        logger.info(f"  {cat_id:>3d}: {actual:>5d} фото")

    with open(os.path.join(output_dir, "scrape_stats.json"), "w") as f:
        json.dump({"downloaded": downloaded, "skipped": skipped, "failed": failed}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Scraper")
    parser.add_argument("--input", nargs="+", required=True, help="JSON файлы с URL")
    parser.add_argument("--output", type=str, default="./dataset", help="Папка для сохранения")
    parser.add_argument("--rate_limit", type=int, default=3, help="Макс запросов/сек")
    parser.add_argument("--max_per_category", type=int, default=500,
                        help="Макс фото на категорию (0 = без лимита)")
    parser.add_argument("--blacklist", type=str, default=None,
                        help="blacklist.json — пропустит мусорные файлы")
    args = parser.parse_args()
    scrape(args.input, args.output, args.rate_limit, args.max_per_category, args.blacklist)