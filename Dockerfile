# NVIDIA Optimized TensorFlow 2.17 + CUDA 12.8 + Blackwell (RTX 5090)
# Последний официальный NGC-контейнер TensorFlow от NVIDIA
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Дополнительные Python-зависимости (TF уже установлен в базовом образе)
RUN pip install --no-cache-dir numpy==1.26.4 Pillow==10.3.0

# Код обучения
COPY train.py .

# Датасет и результаты монтируются снаружи
VOLUME ["/workspace/dataset", "/workspace/output"]

ENTRYPOINT ["python", "train.py"]
CMD ["--help"]
