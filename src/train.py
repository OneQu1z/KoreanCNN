"""
Основной модуль для обучения модели
"""
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.config import (
    MODELS_DIR, DEVICE, CLASS_LABELS, NUM_CLASSES, IMAGE_SIZE, 
    BATCH_SIZE, TRAIN_VAL_SPLIT
)
from src.utils import set_seed, log_info, log_warning
from src.dataset import KoreanAlphabetDataset, get_transforms, get_dataloaders


def main():
    """
    Главная функция обучения
    """
    log_info("=" * 50)
    log_info("Training pipeline initialized")
    log_info("=" * 50)
    
    # Установка seed для воспроизводимости
    set_seed(42)
    log_info("Seed установлен для воспроизводимости результатов")
    
    # Проверка доступности CUDA
    log_info("\nПроверка доступности GPU...")
    if torch.cuda.is_available():
        log_info(f"✓ CUDA доступна!")
        log_info(f"  Устройство: {torch.cuda.get_device_name(0)}")
        log_info(f"  Версия CUDA: {torch.version.cuda}")
    else:
        log_warning("CUDA недоступна, будет использован CPU")
    
    log_info(f"\nИспользуемое устройство: {DEVICE}")
    log_info(f"Директория для сохранения моделей: {MODELS_DIR}")
    
    # Информация о классах
    log_info(f"\nКлассы для распознавания ({NUM_CLASSES}): {', '.join(CLASS_LABELS)}")
    
    # Создание датасета с реальными данными
    log_info("\n" + "-" * 50)
    log_info("Инициализация датасета с изображениями корейских букв...")
    log_info("Генерация через корейские шрифты, установленные в системе")
    log_info("-" * 50)
    
    # Создаем трансформации для предобработки
    transform = get_transforms(is_training=False)
    
    # Создаем датасет (автоматически загрузит или сгенерирует изображения)
    dataset = KoreanAlphabetDataset(transform=transform)
    
    log_info(f"Размер датасета: {len(dataset)} сэмплов")
    log_info(f"Размер изображений: {IMAGE_SIZE}")
    
    # Проверка работы датасета - получаем один сэмпл
    sample_image, sample_label = dataset[0]
    log_info(f"Пример сэмпла: image shape = {sample_image.shape}, label = {dataset.get_class_name(sample_label)} ({sample_label})")
    
    # Разделение на train/validation и создание DataLoader'ов
    log_info("\n" + "-" * 50)
    log_info("Разделение на train/validation и создание DataLoader'ов...")
    log_info("-" * 50)
    
    train_loader, val_loader = get_dataloaders(
        dataset,
        batch_size=BATCH_SIZE,
        train_val_split=TRAIN_VAL_SPLIT,
        seed=42
    )
    
    log_info(f"Размер батча: {BATCH_SIZE}")
    log_info(f"Количество батчей в train: {len(train_loader)}")
    log_info(f"Количество батчей в validation: {len(val_loader)}")
    
    # Получаем один батч и выводим его форму
    log_info("\n" + "-" * 50)
    log_info("Проверка загрузки батча...")
    log_info("-" * 50)
    
    # Получаем один батч из train_loader
    images, labels = next(iter(train_loader))
    
    log_info(f"Форма батча изображений: {images.shape}")
    log_info(f"  - Размер батча: {images.shape[0]}")
    log_info(f"  - Каналы: {images.shape[1]}")
    log_info(f"  - Высота: {images.shape[2]}")
    log_info(f"  - Ширина: {images.shape[3]}")
    log_info(f"Форма батча меток: {labels.shape}")
    log_info(f"Метки в батче: {[dataset.get_class_name(int(label)) for label in labels[:5]]} (первые 5)")
    log_info(f"Диапазон значений пикселей: [{images.min().item():.3f}, {images.max().item():.3f}]")
    
    log_info("\n" + "=" * 50)
    log_info("STEP 3 завершен успешно!")
    log_info("=" * 50)
    
    # Выход с кодом 0 (успешное завершение)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

