"""
Модуль для работы с датасетом корейских букв

Использует генерацию изображений через корейские шрифты, установленные в системе.
"""
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.config import (
    RAW_DATA_DIR, CLASS_LABELS, NUM_CLASSES, 
    IMAGE_SIZE, TRAIN_VAL_SPLIT
)
from src.utils import log_info, log_warning, set_seed
from src.datasets.font_based import FontBasedDataset


class ImageTransform:
    """
    Класс для трансформаций изображений без использования torchvision
    """
    def __init__(self, resize_size, normalize=True):
        """
        Args:
            resize_size (tuple): Размер для изменения размера (height, width)
            normalize (bool): Нормализовать ли изображение в [-1, 1]
        """
        self.resize_size = resize_size
        self.normalize = normalize
    
    def __call__(self, image):
        """
        Применяет трансформации к изображению
        
        Args:
            image: PIL.Image или numpy array
            
        Returns:
            torch.Tensor: Тензор изображения [C, H, W]
        """
        import numpy as np
        from PIL import Image
        
        # Если это PIL Image, конвертируем в grayscale если нужно
        if isinstance(image, Image.Image):
            if image.mode != 'L':
                image = image.convert('L')
            # Изменяем размер
            image = image.resize(self.resize_size[::-1], Image.Resampling.LANCZOS)  # PIL использует (width, height)
        
        # Преобразуем в numpy array
        if isinstance(image, Image.Image):
            img_array = np.array(image, dtype=np.float32)
        else:
            img_array = np.array(image, dtype=np.float32)
        
        # Нормализуем в [0, 1]
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
        
        # Преобразуем в тензор [H, W]
        img_tensor = torch.from_numpy(img_array)
        
        # Добавляем канал [1, H, W] для grayscale
        if len(img_tensor.shape) == 2:
            img_tensor = img_tensor.unsqueeze(0)
        
        # Нормализуем в [-1, 1] если нужно
        if self.normalize:
            img_tensor = (img_tensor - 0.5) / 0.5
        
        return img_tensor


def get_transforms(is_training=False):
    """
    Возвращает трансформации для предобработки изображений
    
    Args:
        is_training (bool): Если True, добавляются аугментации для обучения (TODO)
        
    Returns:
        ImageTransform: Объект трансформации
    """
    return ImageTransform(resize_size=IMAGE_SIZE, normalize=True)


class KoreanAlphabetDataset(Dataset):
    """
    Класс датасета для корейских букв
    
    Генерирует изображения используя корейские шрифты, установленные в системе.
    """
    
    def __init__(self, root_dir=None, transform=None, **kwargs):
        """
        Инициализация датасета
        
        Args:
            root_dir (Path, optional): Корневая директория с данными
            transform (callable, optional): Трансформации для применения к изображениям
            **kwargs: Дополнительные аргументы для FontBasedDataset
        """
        # Создаем датасет на основе шрифтов
        if transform is None:
            transform = get_transforms(is_training=False)
        
        self._dataset = FontBasedDataset(
            root_dir=root_dir,
            transform=transform,
            **kwargs
        )
    
    def __len__(self):
        """Возвращает размер датасета"""
        return len(self._dataset)
    
    def __getitem__(self, idx):
        """Возвращает один сэмпл из датасета"""
        return self._dataset[idx]
    
    def get_class_name(self, class_idx):
        """Возвращает имя класса по индексу"""
        return self._dataset.get_class_name(class_idx)


def get_dataloaders(dataset, batch_size=32, train_val_split=0.8, seed=42):
    """
    Создает DataLoader'ы для обучения и валидации
    
    Args:
        dataset: Полный датасет
        batch_size (int): Размер батча
        train_val_split (float): Доля данных для обучения (остальное для валидации)
        seed (int): Seed для воспроизводимости разделения
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Устанавливаем seed для воспроизводимости разделения
    set_seed(seed)
    
    # Вычисляем размеры
    total_size = len(dataset)
    train_size = int(train_val_split * total_size)
    val_size = total_size - train_size
    
    log_info(f"Разделение датасета: {train_size} для обучения, {val_size} для валидации")
    
    # Разделяем датасет
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Создаем DataLoader'ы
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 0 для Windows совместимости
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader
