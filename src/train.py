"""
Основной модуль для обучения модели
"""
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from src.config import (
    MODELS_DIR, DEVICE, CLASS_LABELS, NUM_CLASSES, IMAGE_SIZE, 
    BATCH_SIZE, TRAIN_VAL_SPLIT, LEARNING_RATE, NUM_EPOCHS, TARGET_ACCURACY
)
from src.utils import set_seed, log_info, log_warning
from src.dataset import KoreanAlphabetDataset, get_transforms, get_dataloaders
from src.model import create_model


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
    
    # Создание модели CNN
    log_info("\n" + "-" * 50)
    log_info("Создание CNN модели...")
    log_info("-" * 50)
    
    model = create_model(num_classes=NUM_CLASSES, input_size=IMAGE_SIZE)
    model = model.to(DEVICE)  # Перемещаем модель на устройство (GPU или CPU)
    
    log_info(f"Модель создана: {model.__class__.__name__}")
    log_info(f"Количество параметров: {model.get_num_parameters():,}")
    log_info(f"Модель размещена на устройстве: {next(model.parameters()).device}")
    
    # Настройка функции потерь и оптимизатора
    log_info("\n" + "-" * 50)
    log_info("Настройка обучения...")
    log_info("-" * 50)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    log_info(f"Функция потерь: CrossEntropyLoss")
    log_info(f"Оптимизатор: Adam (learning rate = {LEARNING_RATE})")
    log_info(f"Максимальное количество эпох: {NUM_EPOCHS}")
    log_info(f"Целевая accuracy: {TARGET_ACCURACY * 100:.0f}%")
    
    # Обучение модели
    best_accuracy = 0.0
    best_model_path = None
    
    for epoch in range(1, NUM_EPOCHS + 1):
        log_info("\n" + "-" * 50)
        log_info(f"Эпоха {epoch}/{NUM_EPOCHS}")
        log_info("-" * 50)
        
        model.train()  # Устанавливаем модель в режим обучения
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Перемещаем данные на устройство
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Обнуляем градиенты
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Вычисляем loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Обновляем веса
            optimizer.step()
            
            # Статистика
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Итоговая статистика за эпоху (train)
        epoch_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        
        # Валидация на validation set
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(DEVICE)
                val_labels = val_labels.to(DEVICE)
                
                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_labels).item()
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        log_info(f"Train - Loss: {epoch_loss:.4f} | Accuracy: {train_accuracy:.2f}% ({correct}/{total})")
        log_info(f"Val   - Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.2f}% ({val_correct}/{val_total})")
        
        # Сохраняем модель, если она лучшая (по validation accuracy)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            
            # Удаляем предыдущую лучшую модель если есть
            if best_model_path and best_model_path.exists():
                best_model_path.unlink()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"korean_cnn_best_epoch{epoch}_train{train_accuracy:.1f}%_val{val_accuracy:.1f}%_{timestamp}.pth"
            best_model_path = MODELS_DIR / model_filename
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_loss,
                'train_accuracy': train_accuracy,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'num_classes': NUM_CLASSES,
                'image_size': IMAGE_SIZE,
            }, best_model_path)
            
            log_info(f"✓ Новая лучшая модель сохранена! (Val Accuracy: {val_accuracy:.2f}%)")
        
        # Проверяем, достигли ли целевой accuracy (по validation)
        if val_accuracy >= TARGET_ACCURACY * 100:
            log_info(f"\n{'='*50}")
            log_info(f"Целевая accuracy достигнута!")
            log_info(f"Validation Accuracy: {val_accuracy:.2f}% >= {TARGET_ACCURACY * 100:.0f}%")
            log_info(f"{'='*50}")
            break
    
    # Финальная информация
    log_info("\n" + "=" * 50)
    log_info("Обучение завершено!")
    log_info(f"Лучшая accuracy: {best_accuracy:.2f}%")
    if best_model_path:
        log_info(f"Лучшая модель сохранена: {best_model_path.name}")
    log_info("=" * 50)
    
    log_info("\n" + "=" * 50)
    log_info("STEP 5 завершен успешно!")
    log_info("=" * 50)
    
    # Выход с кодом 0 (успешное завершение)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

