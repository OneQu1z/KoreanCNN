"""
Модуль для работы с датасетом корейских букв
"""
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CLASS_LABELS, CLASS_TO_IDX, NUM_CLASSES
from src.utils import log_info, log_warning


def download_dataset():
    """
    Функция для загрузки датасета корейских букв
    
    TODO: Реализовать загрузку реального датасета
    Возможные источники:
    - Hangul Fonts Dataset
    - KMNIST (Korean Modified MNIST)
    - Собственный сбор данных
    
    Returns:
        bool: True если датасет успешно загружен, False в противном случае
    """
    log_info("Проверка наличия датасета...")
    
    # Проверяем, существует ли уже обработанный датасет
    if PROCESSED_DATA_DIR.exists() and any(PROCESSED_DATA_DIR.iterdir()):
        log_info("Обработанные данные найдены в data/processed/")
        return True
    
    # Проверяем, существует ли исходный датасет
    if RAW_DATA_DIR.exists() and any(RAW_DATA_DIR.iterdir()):
        log_info("Исходные данные найдены в data/raw/")
        log_warning("Требуется обработка данных. Используются dummy данные.")
        return False
    
    log_warning("Реальный датасет не найден. Используются dummy данные для тестирования.")
    log_warning("TODO: Реализовать загрузку датасета корейских букв")
    return False


class KoreanAlphabetDataset(Dataset):
    """
    Датасет для корейских букв (ㄱ, ㄴ, ㄷ, ㄹ, ㅁ)
    
    Пока что возвращает dummy данные для тестирования структуры проекта.
    TODO: Заменить на реальные данные после загрузки датасета.
    """
    
    def __init__(self, root_dir=None, transform=None, use_real_data=False, num_samples=1000, image_size=(28, 28)):
        """
        Инициализация датасета
        
        Args:
            root_dir (Path, optional): Корневая директория с данными
            transform (callable, optional): Трансформации для применения к изображениям
            use_real_data (bool): Использовать ли реальные данные (если доступны)
            num_samples (int): Количество сэмплов для dummy данных
            image_size (tuple): Размер изображений (height, width)
        """
        self.root_dir = root_dir or RAW_DATA_DIR
        self.transform = transform
        self.use_real_data = use_real_data
        self.num_samples = num_samples
        self.image_size = image_size
        
        # Проверяем доступность реальных данных
        if use_real_data:
            data_available = download_dataset()
            if not data_available:
                log_warning("Реальные данные недоступны, используются dummy данные")
                self.use_real_data = False
        
        if not self.use_real_data:
            log_info(f"Использование dummy данных: {num_samples} сэмплов")
            # TODO: Заменить на реальную загрузку данных
        
        # Генерируем метки для dummy данных (равномерно распределенные по классам)
        samples_per_class = num_samples // NUM_CLASSES
        self.labels = []
        for class_idx in range(NUM_CLASSES):
            self.labels.extend([class_idx] * samples_per_class)
        
        # Добавляем оставшиеся сэмплы к первому классу для точного количества
        remaining = num_samples - len(self.labels)
        self.labels.extend([0] * remaining)
        
        log_info(f"Датасет инициализирован: {len(self.labels)} сэмплов, {NUM_CLASSES} классов")
    
    def __len__(self):
        """
        Возвращает размер датасета
        
        Returns:
            int: Количество сэмплов в датасете
        """
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Возвращает один сэмпл из датасета
        
        Args:
            idx (int): Индекс сэмпла
            
        Returns:
            tuple: (image, label) где image - тензор изображения, label - индекс класса
        """
        if self.use_real_data:
            # TODO: Реализовать загрузку реального изображения
            # image = load_image_from_disk(idx)
            # label = self.labels[idx]
            pass
        
        # Генерируем dummy изображение (случайные пиксели)
        # В реальном датасете это будет загруженное изображение
        image = np.random.rand(*self.image_size).astype(np.float32)
        
        # Применяем трансформации если они указаны
        if self.transform:
            image = self.transform(image)
        else:
            # Преобразуем в тензор если трансформаций нет
            image = torch.from_numpy(image)
            # Добавляем канал (1 channel для grayscale)
            image = image.unsqueeze(0)
        
        label = self.labels[idx]
        
        return image, label
    
    def get_class_name(self, class_idx):
        """
        Возвращает имя класса по индексу
        
        Args:
            class_idx (int): Индекс класса
            
        Returns:
            str: Метка класса (например, "ㄱ")
        """
        return CLASS_LABELS[class_idx]
