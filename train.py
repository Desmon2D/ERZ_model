"""
=============================================================================
Violation Classifier v2 — TensorFlow Training Pipeline
=============================================================================
Архитектура: EfficientNetB2 (transfer learning с ImageNet)
Улучшения: Focal Loss, Label Smoothing, Top-3 метрика
Выход: TFLite INT8 квантизованная модель

Использование:
    python train.py --data_dir ./dataset --epochs_phase1 20 --epochs_phase2 50 --batch_size 32 --focal_loss --use_class_weights
=============================================================================
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
)
from datetime import datetime


# =============================================================================
# 1. КОНФИГУРАЦИЯ
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Violation Classifier v2")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--img_size", type=int, default=260)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs_phase1", type=int, default=20)
    parser.add_argument("--epochs_phase2", type=int, default=50)
    parser.add_argument("--lr_phase1", type=float, default=1e-3)
    parser.add_argument("--lr_phase2", type=float, default=1e-5)
    parser.add_argument("--dropout_rate", type=float, default=0.4)
    parser.add_argument("--fine_tune_from", type=int, default=200)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--subfolder", type=str, default="d")
    parser.add_argument("--include_normal", action="store_true")
    parser.add_argument("--normal_subfolder", type=str, default="f")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--focal_loss", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--backbone", type=str, default="efficientnetb2",
                        choices=["efficientnetb0", "efficientnetb2", "mobilenetv2"])
    return parser.parse_args()


# =============================================================================
# 2. FOCAL LOSS
# =============================================================================

class SparseFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, label_smoothing=0.0, num_classes=114, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        y_true_oh = tf.one_hot(y_true, self.num_classes)
        if self.label_smoothing > 0:
            y_true_oh = y_true_oh * (1.0 - self.label_smoothing) + \
                        self.label_smoothing / self.num_classes
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true_oh * tf.math.log(y_pred)
        p_t = tf.reduce_sum(y_pred * y_true_oh, axis=-1, keepdims=True)
        weight = tf.pow(1.0 - p_t, self.gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=-1))


# =============================================================================
# 3. TOP-K METRIC
# =============================================================================

class SparseTopKAccuracy(tf.keras.metrics.Metric):
    def __init__(self, k=3, name="top3_acc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = k
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        top_k = tf.math.top_k(y_pred, k=self.k).indices
        matches = tf.reduce_any(tf.equal(top_k, tf.expand_dims(y_true, -1)), axis=-1)
        self.correct.assign_add(tf.reduce_sum(tf.cast(matches, tf.float32)))
        self.total.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.correct / (self.total + 1e-7)

    def reset_state(self):
        self.correct.assign(0)
        self.total.assign(0)


# =============================================================================
# 4. ДАННЫЕ
# =============================================================================

def discover_categories(data_dir, subfolder):
    categories = {}
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for entry in sorted(os.listdir(data_dir)):
        entry_path = os.path.join(data_dir, entry)
        if not os.path.isdir(entry_path) or entry.startswith("_"):
            continue
        try:
            cat_id = int(entry)
        except ValueError:
            continue
        sub_path = os.path.join(entry_path, subfolder)
        if not os.path.exists(sub_path):
            continue
        files = [os.path.join(sub_path, f) for f in os.listdir(sub_path)
                 if os.path.splitext(f)[1].lower() in valid_ext]
        if files:
            categories[cat_id] = files
    return categories


def prepare_splits(categories, val_split, test_split, seed):
    np.random.seed(seed)
    sorted_cats = sorted(categories.keys())
    label_map = {cat_id: idx for idx, cat_id in enumerate(sorted_cats)}
    train_p, train_l, val_p, val_l, test_p, test_l = [], [], [], [], [], []

    for cat_id in sorted_cats:
        files = list(categories[cat_id])
        np.random.shuffle(files)
        n = len(files)
        idx = label_map[cat_id]

        if n == 1:
            train_p.append(files[0]); train_l.append(idx); continue
        if n == 2:
            train_p.append(files[0]); train_l.append(idx)
            val_p.append(files[1]); val_l.append(idx); continue

        n_test = max(1, int(n * test_split))
        n_val = max(1, int(n * val_split))
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train = 1; rem = n - 1
            n_val = min(max(1, rem // 2), rem); n_test = rem - n_val

        s1, s2 = n_train, n_train + n_val
        for f in files[:s1]: train_p.append(f); train_l.append(idx)
        for f in files[s1:s2]: val_p.append(f); val_l.append(idx)
        for f in files[s2:s2+n_test]: test_p.append(f); test_l.append(idx)

    return (train_p, train_l), (val_p, val_l), (test_p, test_l), label_map


def create_tf_dataset(paths, labels, img_size, batch_size, is_training, preprocess_fn, seed=42):
    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        img = preprocess_fn(img)
        return img, label

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_saturation(img, 0.8, 1.2)
        img = tf.image.random_crop(
            tf.image.resize(img, [img_size + 30, img_size + 30]),
            [img_size, img_size, 3])
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if is_training:
        ds = ds.shuffle(len(paths), seed=seed)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def compute_class_weights(labels):
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    n = len(unique)
    return {int(c): total / (n * cnt) for c, cnt in zip(unique, counts)}


# =============================================================================
# 5. МОДЕЛЬ
# =============================================================================

def get_backbone(name, img_size):
    if name == "efficientnetb2":
        base = tf.keras.applications.EfficientNetB2(
            input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet")
        prep = tf.keras.applications.efficientnet.preprocess_input
    elif name == "efficientnetb0":
        base = tf.keras.applications.EfficientNetB0(
            input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet")
        prep = tf.keras.applications.efficientnet.preprocess_input
    else:
        base = tf.keras.applications.MobileNetV2(
            input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet")
        prep = tf.keras.applications.mobilenet_v2.preprocess_input
    return base, prep


def build_model(num_classes, img_size, dropout_rate, backbone_name):
    base, prep = get_backbone(backbone_name, img_size)
    base.trainable = False
    inp = layers.Input(shape=(img_size, img_size, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inp, out), base, prep


# =============================================================================
# 6. TFLITE
# =============================================================================

def convert_to_tflite(model, train_paths, img_size, run_dir, prep_fn):
    def rep_data():
        for p in train_paths[:200]:
            img = tf.io.read_file(p)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [img_size, img_size])
            img = prep_fn(img)
            yield [tf.expand_dims(img, 0)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_data
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.uint8

    print("  Конвертация TFLite INT8...")
    tfl = conv.convert()
    path = os.path.join(run_dir, "model_int8.tflite")
    with open(path, "wb") as f:
        f.write(tfl)
    mb = os.path.getsize(path) / 1024 / 1024
    print(f"  Сохранено: {path} ({mb:.2f} MB)")


# =============================================================================
# 7. MAIN
# =============================================================================

def train(args):
    print("=" * 60)
    print("Violation Classifier v2")
    print(f"  Backbone:        {args.backbone}")
    print(f"  Focal Loss:      {args.focal_loss} (gamma={args.focal_gamma})")
    print(f"  Label Smoothing: {args.label_smoothing}")
    print(f"  Image Size:      {args.img_size}")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Данные
    print(f"\n[1/6] Сканирование: {args.data_dir}")
    cats = discover_categories(args.data_dir, args.subfolder)

    if args.include_normal:
        nf = []
        vext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for e in sorted(os.listdir(args.data_dir)):
            ep = os.path.join(args.data_dir, e)
            if not os.path.isdir(ep) or e.startswith("_"): continue
            sp = os.path.join(ep, args.normal_subfolder)
            if not os.path.exists(sp): continue
            for f in os.listdir(sp):
                if os.path.splitext(f)[1].lower() in vext:
                    nf.append(os.path.join(sp, f))
        if nf:
            cats[0] = nf
            print(f"  Класс «Норма»: {len(nf)} фото")

    nc = len(cats)
    ti = sum(len(v) for v in cats.values())
    print(f"  Категорий: {nc}, изображений: {ti}")
    for c in sorted(cats.keys()):
        m = " !!!" if len(cats[c]) < 20 else ""
        print(f"    {c:>3d}: {len(cats[c]):>5d}{m}")

    if nc == 0:
        print("ОШИБКА: нет категорий!"); return

    # Разбиение
    print(f"\n[2/6] Разбиение")
    (tp, tl), (vp, vl), (ep, el), lm = prepare_splits(cats, args.val_split, args.test_split, args.seed)
    print(f"  Train: {len(tp)} | Val: {len(vp)} | Test: {len(ep)}")
    im = {v: k for k, v in lm.items()}
    with open(os.path.join(run_dir, "label_map.json"), "w") as f:
        json.dump(im, f, indent=2)

    # Модель
    print(f"\n[3/6] Модель: {args.backbone}")
    model, base, prep = build_model(nc, args.img_size, args.dropout_rate, args.backbone)

    # Датасеты
    print(f"\n[4/6] Датасеты")
    tds = create_tf_dataset(tp, tl, args.img_size, args.batch_size, True, prep, args.seed)
    vds = create_tf_dataset(vp, vl, args.img_size, args.batch_size, False, prep)
    eds = create_tf_dataset(ep, el, args.img_size, args.batch_size, False, prep)

    cw = compute_class_weights(tl) if args.use_class_weights else None

    # Loss
    if args.focal_loss:
        loss = SparseFocalLoss(args.focal_gamma, args.label_smoothing, nc)
    else:
        loss = "sparse_categorical_crossentropy"

    metrics = ["accuracy", SparseTopKAccuracy(k=3)]

    cbs = [
        EarlyStopping(monitor="val_top3_acc", patience=10, mode="max",
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint(os.path.join(run_dir, "best_model.keras"),
                        monitor="val_top3_acc", mode="max", save_best_only=True, verbose=1),
        TensorBoard(log_dir=os.path.join(run_dir, "logs")),
    ]

    model.summary()

    # Phase 1
    print(f"\n[5/6] Phase 1: Feature Extraction ({args.epochs_phase1} эпох)")
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr_phase1), loss=loss, metrics=metrics)
    model.fit(tds, validation_data=vds, epochs=args.epochs_phase1, callbacks=cbs, class_weight=cw)

    # Phase 2
    print(f"\n[6/6] Phase 2: Fine-Tuning ({args.epochs_phase2} эпох)")
    base.trainable = True
    for l in base.layers[:args.fine_tune_from]:
        l.trainable = False
    print(f"  Обучаемых слоёв: {sum(1 for l in base.layers if l.trainable)}/{len(base.layers)}")
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr_phase2), loss=loss, metrics=metrics)
    model.fit(tds, validation_data=vds, epochs=args.epochs_phase2, callbacks=cbs, class_weight=cw)

    # Тест
    print("\n" + "=" * 60)
    r = model.evaluate(eds, return_dict=True)
    print(f"  Test Top-1: {r['accuracy']:.4f}")
    print(f"  Test Top-3: {r['top3_acc']:.4f}")

    # Сохранение
    model.save(os.path.join(run_dir, "final_model.keras"))

    # TFLite
    print("\n" + "=" * 60)
    convert_to_tflite(model, tp, args.img_size, run_dir, prep)

    # Метаданные
    meta = {
        "num_classes": nc, "backbone": args.backbone, "img_size": args.img_size,
        "focal_loss": args.focal_loss, "label_smoothing": args.label_smoothing,
        "test_top1": float(r["accuracy"]), "test_top3": float(r["top3_acc"]),
        "label_map": im, "timestamp": timestamp, "args": vars(args),
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"ГОТОВО! Top-1: {r['accuracy']:.1%} | Top-3: {r['top3_acc']:.1%}")
    print(f"  {run_dir}")


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU: {gpus}")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    else:
        print("GPU не найден")
    train(parse_args())