"""
Batch test: 10 images per category, top-3 predictions, accuracy stats.
"""

import os
import json
import random
import numpy as np
from PIL import Image

# Suppress TF warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

MODEL_PATH = "./output/run_20260319_083744/model_int8.tflite"
LABELS_PATH = "./output/run_20260319_083744/label_map.json"
DATASET_DIR = "./dataset"
SAMPLES_PER_CAT = 10
IMG_SIZE = 224
SEED = 42


def main():
    random.seed(SEED)

    # Load model
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load label map (class_index -> category_id)
    with open(LABELS_PATH) as f:
        label_map = json.load(f)

    # Discover categories
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    categories = {}
    for entry in sorted(os.listdir(DATASET_DIR)):
        d_path = os.path.join(DATASET_DIR, entry, "d")
        if not os.path.isdir(d_path):
            continue
        try:
            cat_id = int(entry)
        except ValueError:
            continue
        files = [os.path.join(d_path, f) for f in os.listdir(d_path)
                 if os.path.splitext(f)[1].lower() in valid_ext]
        if files:
            categories[cat_id] = files

    # Run inference
    results = {}  # cat_id -> {total, correct, top3_correct, predictions}

    total_cats = len(categories)
    for i, cat_id in enumerate(sorted(categories.keys()), 1):
        files = categories[cat_id]
        sample = random.sample(files, min(SAMPLES_PER_CAT, len(files)))
        total_in_dataset = len(files)

        correct = 0
        top3_correct = 0
        all_preds = []

        for img_path in sample:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
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

        # Aggregate top-3 predictions (most common top-1)
        from collections import Counter
        top1_counts = Counter(p[0][0] for p in all_preds)
        most_common_top1 = top1_counts.most_common(3)

        # Average probabilities for top-3 across samples
        avg_probs = np.mean([p[1] for p in all_preds], axis=0)
        avg_top3_cats_idx = np.argsort(np.mean(
            [probs for _, probs in [(None, None)] * 0], axis=0) if False else avg_probs)

        # Just use average top-3
        avg_all = np.mean([[probs[i] for i in np.argsort(
            [float(output[0][j]) for j in range(len(output[0]))])[::-1][:3]]
            for _ in [None]], axis=0) if False else None

        n_tested = len(sample)
        results[cat_id] = {
            "total_in_dataset": total_in_dataset,
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

        # Get aggregated top-3 from averaging all sample predictions
        # Collect all top-1 predictions with their average confidence
        from collections import defaultdict
        cat_probs = defaultdict(list)
        for top3_cats, top3_probs in r["sample_preds"]:
            for c, p in zip(top3_cats, top3_probs):
                cat_probs[c].append(p)

        # Sort by frequency then by avg probability
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
