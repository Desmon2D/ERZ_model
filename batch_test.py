"""
Batch test: проверка модели на датасете с расчётом accuracy.

Использование:
    # Указать папку с моделью (model_int8.tflite + label_map.json):
    python batch_test.py --run_dir ./output/run_20260325_084645

    # Или указать файлы явно:
    python batch_test.py --model ./output/run_20260325_084645/model_int8.tflite \
                         --labels ./output/run_20260325_084645/label_map.json

    # Тест только определённых категорий:
    python batch_test.py --run_dir ./output/run_20260325_084645 --categories 1 5 12 99

    # Указать папку с данными и кол-во сэмплов:
    python batch_test.py --run_dir ./output/run_20260325_084645 \
                         --dataset_dir ./dataset --samples 20
"""

import argparse
import os
import json
import random
import numpy as np
from PIL import Image
from collections import Counter, defaultdict

# Suppress TF warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_model(model_path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
    interpreter = Interpreter(model_path=model_path, num_threads=os.cpu_count())
    interpreter.allocate_tensors()
    return interpreter


def main():
    parser = argparse.ArgumentParser(description="Batch test: accuracy по категориям")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Папка с результатами обучения (содержит model_int8.tflite и label_map.json)")
    parser.add_argument("--model", type=str, default=None,
                        help="Путь к model_int8.tflite")
    parser.add_argument("--labels", type=str, default=None,
                        help="Путь к label_map.json")
    parser.add_argument("--dataset_dir", type=str, default="./dataset",
                        help="Папка с датасетом (default: ./dataset)")
    parser.add_argument("--categories", type=int, nargs="+", default=None,
                        help="Список категорий для тестирования (по умолчанию — все)")
    parser.add_argument("--samples", type=int, default=10,
                        help="Количество изображений на категорию (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    # Resolve model and labels paths
    if args.run_dir:
        model_path = args.model or os.path.join(args.run_dir, "model_int8.tflite")
        labels_path = args.labels or os.path.join(args.run_dir, "label_map.json")
    elif args.model and args.labels:
        model_path = args.model
        labels_path = args.labels
    else:
        parser.error("Укажите --run_dir или оба --model и --labels")

    if not os.path.isfile(model_path):
        parser.error(f"Модель не найдена: {model_path}")
    if not os.path.isfile(labels_path):
        parser.error(f"Label map не найден: {labels_path}")

    random.seed(args.seed)

    # Load model
    interpreter = load_model(model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Auto-detect image size from model
    img_size = input_details[0]["shape"][1]
    print(f"Модель: {model_path}")
    print(f"IMG_SIZE: {img_size} (из модели)")
    print(f"Датасет: {args.dataset_dir}")
    print()

    # Load label map
    with open(labels_path) as f:
        label_map = json.load(f)

    # Discover categories
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    categories = {}
    for entry in sorted(os.listdir(args.dataset_dir)):
        d_path = os.path.join(args.dataset_dir, entry, "d")
        if not os.path.isdir(d_path):
            continue
        try:
            cat_id = int(entry)
        except ValueError:
            continue
        # Filter by requested categories
        if args.categories and cat_id not in args.categories:
            continue
        files = [os.path.join(d_path, f) for f in os.listdir(d_path)
                 if os.path.splitext(f)[1].lower() in valid_ext]
        if files:
            categories[cat_id] = files

    if not categories:
        print("Категории не найдены!")
        return

    print(f"Категорий: {len(categories)}, сэмплов/кат: {args.samples}")
    print()

    # Run inference
    results = {}
    total_cats = len(categories)

    for i, cat_id in enumerate(sorted(categories.keys()), 1):
        files = categories[cat_id]
        sample = random.sample(files, min(args.samples, len(files)))

        correct = 0
        top3_correct = 0
        all_preds = []

        for img_path in sample:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((img_size, img_size), Image.BILINEAR)
            img_array = np.expand_dims(np.array(img, dtype=np.uint8), 0)

            interpreter.set_tensor(input_details[0]["index"], img_array)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]["index"])

            if output_details[0]["dtype"] == np.uint8:
                scale, zero_point = output_details[0]["quantization"]
                output = (output.astype(np.float32) - zero_point) * scale

            probs = output[0]
            top3_idx = np.argsort(probs)[::-1][:3]
            top3_cats = [int(label_map.get(str(idx), idx)) for idx in top3_idx]
            top3_probs = [float(probs[idx]) for idx in top3_idx]

            if top3_cats[0] == cat_id:
                correct += 1
            if cat_id in top3_cats:
                top3_correct += 1

            all_preds.append((top3_cats, top3_probs))

        n_tested = len(sample)
        top1_counts = Counter(p[0][0] for p in all_preds)
        most_common_top1 = top1_counts.most_common(3)

        results[cat_id] = {
            "total_in_dataset": len(files),
            "n_tested": n_tested,
            "top1_acc": correct / n_tested,
            "top3_acc": top3_correct / n_tested,
            "most_common": most_common_top1,
            "sample_preds": all_preds,
        }

        print(f"\r  Processing: {i}/{total_cats} categories...", end="", flush=True)

    print("\r" + " " * 50 + "\r", end="")

    # Print table
    print(f"{'Cat':>5} | {'Photos':>6} | {'Test':>4} | {'Top1':>5} | {'Top3':>5} | "
          f"{'Pred-1':>20} | {'Pred-2':>20} | {'Pred-3':>20}")
    print("-" * 115)

    total_correct = 0
    total_top3 = 0
    total_tested = 0

    for cat_id in sorted(results.keys()):
        r = results[cat_id]
        n = r["n_tested"]
        total_correct += int(r["top1_acc"] * n)
        total_top3 += int(r["top3_acc"] * n)
        total_tested += n

        cat_probs = defaultdict(list)
        for top3_cats, top3_probs in r["sample_preds"]:
            for c, p in zip(top3_cats, top3_probs):
                cat_probs[c].append(p)

        sorted_preds = sorted(cat_probs.items(),
                              key=lambda x: (len(x[1]), np.mean(x[1])),
                              reverse=True)[:3]

        pred_strs = []
        for pred_cat, probs_list in sorted_preds:
            avg_p = np.mean(probs_list)
            freq = len(probs_list)
            marker = "*" if pred_cat == cat_id else " "
            pred_strs.append(f"{marker}{pred_cat:>3d} {avg_p:5.1%} ({freq:>2d}x)")

        while len(pred_strs) < 3:
            pred_strs.append(" " * 20)

        ok = "OK" if r["top1_acc"] >= 0.5 else "--"

        print(f"{cat_id:>5} | {r['total_in_dataset']:>6} | {n:>4} | "
              f"{r['top1_acc']:>4.0%} {ok} | {r['top3_acc']:>4.0%}  | "
              f"{pred_strs[0]:>20} | {pred_strs[1]:>20} | {pred_strs[2]:>20}")

    print("-" * 115)
    print(f"{'TOTAL':>5} | {'':>6} | {total_tested:>4} | "
          f"{total_correct/total_tested:>4.0%}    | {total_top3/total_tested:>4.0%}  |")


if __name__ == "__main__":
    main()
