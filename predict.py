"""
Инференс: классификация нарушения по фотографии.
Использует INT8 TFLite модель.

Использование:
    # Одно изображение:
    python predict.py --model ./output/run_20260325_084645/model_int8.tflite \
                      --labels ./output/run_20260325_084645/label_map.json \
                      --image photo.jpg

    # Папка с изображениями:
    python predict.py --model ./output/run_20260325_084645/model_int8.tflite \
                      --labels ./output/run_20260325_084645/label_map.json \
                      --image_dir ./photos/ --top_k 3

    # Несколько конкретных файлов:
    python predict.py --model ./output/run_20260325_084645/model_int8.tflite \
                      --labels ./output/run_20260325_084645/label_map.json \
                      --image photo1.jpg photo2.jpg photo3.jpg
"""

import argparse
import json
import os
import sys
import numpy as np
from PIL import Image


def load_model(model_path: str):
    """Загружает TFLite модель и возвращает интерпретатор."""
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


def predict(interpreter, image_path: str, img_size: int = 224) -> np.ndarray:
    """Запускает инференс на одном изображении. Возвращает массив вероятностей."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Загрузка и подготовка изображения
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    img_array = np.array(img, dtype=np.uint8)
    img_array = np.expand_dims(img_array, axis=0)

    # Инференс
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    # Если выход uint8 — преобразуем в вероятности через quantization params
    if output_details[0]["dtype"] == np.uint8:
        scale, zero_point = output_details[0]["quantization"]
        output = (output.astype(np.float32) - zero_point) * scale

    return output[0]


def format_results(probabilities: np.ndarray, label_map: dict, top_k: int = 5):
    """Возвращает top-k предсказаний в виде списка (category_id, probability)."""
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    results = []
    for idx in top_indices:
        cat_id = label_map.get(str(idx), idx)
        prob = float(probabilities[idx])
        results.append((cat_id, prob))
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict violation class from photo")
    parser.add_argument("--model", type=str, required=True,
                        help="Путь к model_int8.tflite")
    parser.add_argument("--labels", type=str, required=True,
                        help="Путь к label_map.json")
    parser.add_argument("--image", type=str, nargs="+", default=None,
                        help="Путь к одному или нескольким изображениям")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Папка с изображениями для batch-предсказания")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Количество top предсказаний")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Укажите --image или --image_dir")

    # Загрузка модели и меток
    interpreter = load_model(args.model)
    with open(args.labels, "r") as f:
        label_map = json.load(f)

    # Определяем img_size из модели
    input_details = interpreter.get_input_details()
    img_size = input_details[0]["shape"][1]

    # Собираем список файлов
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = []

    if args.image:
        image_paths.extend(args.image)
    if args.image_dir:
        for fname in sorted(os.listdir(args.image_dir)):
            if os.path.splitext(fname)[1].lower() in valid_ext:
                image_paths.append(os.path.join(args.image_dir, fname))

    if not image_paths:
        print("No images found.")
        sys.exit(1)

    # Инференс
    for path in image_paths:
        probs = predict(interpreter, path, img_size)
        results = format_results(probs, label_map, args.top_k)

        print(f"\n{path}")
        for rank, (cat_id, prob) in enumerate(results, 1):
            bar = "#" * int(prob * 30)
            print(f"  {rank}. Category {cat_id:>3d}  {prob:6.2%}  {bar}")


if __name__ == "__main__":
    main()
