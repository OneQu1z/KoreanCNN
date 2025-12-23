"""
Основной модуль для обучения модели
"""
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.config import MODELS_DIR, DEVICE, CLASS_LABELS, NUM_CLASSES, IMAGE_SIZE, DEFAULT_NUM_SAMPLES
from src.utils import set_seed, log_info, log_warning
from src.dataset import KoreanAlphabetDataset


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
    
    # Создание датасета
    log_info("\n" + "-" * 50)
    log_info("Инициализация датасета...")
    log_info("-" * 50)
    
    dataset = KoreanAlphabetDataset(
        use_real_data=False,  # TODO: Изменить на True когда реальный датасет будет доступен
        num_samples=DEFAULT_NUM_SAMPLES,
        image_size=IMAGE_SIZE
    )
    
    log_info(f"Размер датасета: {len(dataset)} сэмплов")
    log_info(f"Размер изображений: {IMAGE_SIZE}")
    
    # Проверка работы датасета - получаем один сэмпл
    sample_image, sample_label = dataset[0]
    log_info(f"Пример сэмпла: image shape = {sample_image.shape}, label = {dataset.get_class_name(sample_label)} ({sample_label})")
    
    log_info("\n" + "=" * 50)
    log_info("STEP 2 завершен успешно!")
    log_info("=" * 50)
    
    # Выход с кодом 0 (успешное завершение)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

