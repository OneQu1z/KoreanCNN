"""
Модуль для предсказаний на обученной модели
"""
import sys
from pathlib import Path
import argparse

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from src.config import (
    MODELS_DIR, DEVICE, CLASS_LABELS, NUM_CLASSES, IMAGE_SIZE, IDX_TO_CLASS
)
from src.utils import log_info, log_warning, log_error
from src.model import create_model
from src.dataset import get_transforms


def load_model(model_path, device=DEVICE):
    """
    Загружает обученную модель из файла
    
    Args:
        model_path (Path): Путь к файлу модели (.pth)
        device (torch.device): Устройство для загрузки модели
        
    Returns:
        tuple: (model, checkpoint_info) - модель и информация о checkpoint
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    
    log_info(f"Загрузка модели из: {model_path}")
    
    # Загружаем checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Извлекаем параметры из checkpoint
    num_classes = checkpoint.get('num_classes', NUM_CLASSES)
    image_size = checkpoint.get('image_size', IMAGE_SIZE)
    
    # Создаем модель с правильными параметрами
    model = create_model(num_classes=num_classes, input_size=image_size)
    
    # Загружаем веса
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # Устанавливаем в режим оценки
    
    log_info(f"Модель загружена успешно")
    log_info(f"  - Количество классов: {num_classes}")
    log_info(f"  - Размер изображения: {image_size}")
    if 'epoch' in checkpoint:
        log_info(f"  - Эпоха обучения: {checkpoint['epoch']}")
    if 'val_accuracy' in checkpoint:
        log_info(f"  - Validation Accuracy: {checkpoint['val_accuracy']:.2f}%")
    
    return model, checkpoint


def preprocess_image(image_path, image_size=IMAGE_SIZE):
    """
    Загружает и предобрабатывает изображение для модели
    
    Args:
        image_path (Path): Путь к изображению
        image_size (tuple): Размер изображения (height, width)
        
    Returns:
        torch.Tensor: Предобработанное изображение [1, 1, height, width]
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")
    
    # Загружаем изображение
    try:
        img = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Не удалось загрузить изображение: {e}")
    
    # Применяем трансформации (как в dataset.py)
    transform = get_transforms(is_training=False)
    img_tensor = transform(img)
    
    # Добавляем batch dimension: [1, height, width] -> [1, 1, height, width]
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor


def predict(model, image_tensor, device=DEVICE):
    """
    Делает предсказание на изображении
    
    Args:
        model: Обученная модель
        image_tensor (torch.Tensor): Предобработанное изображение [1, 1, H, W]
        device (torch.device): Устройство для вычислений
        
    Returns:
        tuple: (predicted_class, confidence, all_probs) - предсказанный класс, уверенность, все вероятности
    """
    # Перемещаем на устройство
    image_tensor = image_tensor.to(device)
    
    # Делаем предсказание
    with torch.no_grad():
        outputs = model(image_tensor)
        
        # Применяем softmax для получения вероятностей
        probs = F.softmax(outputs, dim=1)
        
        # Получаем предсказанный класс и уверенность
        confidence, predicted_idx = torch.max(probs, 1)
        predicted_idx = predicted_idx.item()
        confidence = confidence.item()
        
        # Получаем все вероятности
        all_probs = probs[0].cpu().numpy()
    
    # Преобразуем индекс в метку класса
    predicted_class = IDX_TO_CLASS[predicted_idx]
    
    return predicted_class, confidence, all_probs


def find_latest_model(models_dir=MODELS_DIR):
    """
    Находит последнюю сохраненную модель
    
    Args:
        models_dir (Path): Директория с моделями
        
    Returns:
        Path: Путь к последней модели или None
    """
    model_files = list(models_dir.glob("*.pth"))
    
    if not model_files:
        return None
    
    # Сортируем по времени модификации (последняя - самая новая)
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return model_files[0]


def main():
    """
    Главная функция для предсказаний
    """
    parser = argparse.ArgumentParser(
        description='Предсказание корейской буквы на изображении',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python -m src.predict --image data/raw/ㄱ/ㄱ_NanumGothic-Regular_0000.png
  python -m src.predict --image path/to/image.png --model models/korean_cnn_best_epoch9_train99.8%_val100.0%_20251223_204142.pth
        """
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        required=True,
        help='Путь к изображению для предсказания'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Путь к модели (если не указан, используется последняя сохраненная)'
    )
    
    args = parser.parse_args()
    
    # Преобразуем пути
    image_path = Path(args.image)
    if args.model:
        model_path = Path(args.model)
    else:
        # Ищем последнюю модель
        model_path = find_latest_model()
        if model_path is None:
            log_error("Не найдено ни одной модели в директории models/")
            log_error("Сначала обучите модель: python -m src.train")
            return 1
        log_info(f"Используется последняя модель: {model_path.name}")
    
    # Проверки
    if not image_path.exists():
        log_error(f"Изображение не найдено: {image_path}")
        return 1
    
    if not model_path.exists():
        log_error(f"Модель не найдена: {model_path}")
        return 1
    
    try:
        # Загружаем модель
        log_info("=" * 50)
        log_info("Загрузка модели...")
        log_info("=" * 50)
        model, checkpoint = load_model(model_path, DEVICE)
        
        # Загружаем и предобрабатываем изображение
        log_info("\n" + "-" * 50)
        log_info("Предобработка изображения...")
        log_info("-" * 50)
        log_info(f"Путь к изображению: {image_path}")
        image_tensor = preprocess_image(image_path)
        log_info(f"Размер изображения после предобработки: {image_tensor.shape}")
        
        # Делаем предсказание
        log_info("\n" + "-" * 50)
        log_info("Выполнение предсказания...")
        log_info("-" * 50)
        predicted_class, confidence, all_probs = predict(model, image_tensor, DEVICE)
        
        # Выводим результаты
        log_info("\n" + "=" * 50)
        log_info("РЕЗУЛЬТАТЫ ПРЕДСКАЗАНИЯ")
        log_info("=" * 50)
        log_info(f"Предсказанная буква: {predicted_class}")
        log_info(f"Уверенность: {confidence * 100:.2f}%")
        log_info("\nВероятности для всех классов:")
        for idx, (class_label, prob) in enumerate(zip(CLASS_LABELS, all_probs)):
            marker = " ←" if class_label == predicted_class else ""
            log_info(f"  {class_label}: {prob * 100:6.2f}%{marker}")
        
        log_info("\n" + "=" * 50)
        
        return 0
        
    except Exception as e:
        log_error(f"Ошибка при выполнении предсказания: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
