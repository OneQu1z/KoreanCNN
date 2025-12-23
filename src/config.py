"""
Конфигурация проекта
"""
import os
import torch
from pathlib import Path

# Базовые пути проекта
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Создаем директории если их нет
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Метки классов корейских букв
CLASS_LABELS = ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ"]
NUM_CLASSES = len(CLASS_LABELS)

# Словарь для обратного преобразования (индекс -> метка)
CLASS_TO_IDX = {label: idx for idx, label in enumerate(CLASS_LABELS)}
IDX_TO_CLASS = {idx: label for label, idx in CLASS_TO_IDX.items()}


def get_device():
    """
    Определяет доступное устройство для вычислений (CUDA или CPU)
    
    Returns:
        torch.device: Устройство для выполнения операций PyTorch
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# Глобальная переменная для устройства (можно переопределить при необходимости)
DEVICE = get_device()

# Параметры датасета
IMAGE_SIZE = (28, 28)  # Размер изображений (height, width) - можно адаптировать под датасет
DEFAULT_NUM_SAMPLES = 1000  # Количество сэмплов для dummy данных

# Параметры разделения данных
TRAIN_VAL_SPLIT = 0.8  # 80% для обучения, 20% для валидации
BATCH_SIZE = 32  # Размер батча для обучения

