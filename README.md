# ERZ Model — Violation Classifier

EfficientNetB2 (transfer learning) для классификации нарушений по фотографиям. INT8 TFLite модель для деплоя.

## Структура

```
train.py          # Обучение модели
predict.py        # Инференс на фото
batch_test.py     # Пакетная проверка accuracy
Dockerfile        # NVIDIA TF 2.17 + CUDA
docker-compose.yml # Сервисы train / test
```

## Обучение

### Docker (GPU)

```bash
docker compose up train
```

Параметры по умолчанию — в `docker-compose.yml`. Или напрямую:

```bash
docker compose run train \
  --data_dir /workspace/dataset \
  --output_dir /workspace/output \
  --epochs_phase1 20 --epochs_phase2 50 \
  --batch_size 64 --focal_loss --use_class_weights
```

### Локально

```bash
python train.py --data_dir ./dataset --epochs_phase1 20 --epochs_phase2 50 --batch_size 32 --focal_loss --use_class_weights
```

Результат: `output/run_<timestamp>/` — `model_int8.tflite`, `label_map.json`, `metadata.json`.

## Инференс

```bash
# Одно изображение
python predict.py --model ./output/run_*/model_int8.tflite \
                  --labels ./output/run_*/label_map.json \
                  --image photo.jpg

# Несколько файлов
python predict.py --model ... --labels ... --image photo1.jpg photo2.jpg

# Папка
python predict.py --model ... --labels ... --image_dir ./photos/ --top_k 3
```

Зависимости (без TF): `pip install ai-edge-litert numpy Pillow`

## Пакетный тест

```bash
# Через run_dir (автоматически находит model + labels)
python batch_test.py --run_dir ./output/run_20260325_084645 \
                     --dataset_dir ./dataset --samples 20

# Или указать файлы явно
python batch_test.py --model ./output/run_*/model_int8.tflite \
                     --labels ./output/run_*/label_map.json

# Только определённые категории
python batch_test.py --run_dir ... --categories 1 5 12 99
```

### Docker (GPU)

```bash
docker compose up test
```

## Датасет

Структура: `dataset/<category_id>/d/<images>`. Категории — числовые ID.
