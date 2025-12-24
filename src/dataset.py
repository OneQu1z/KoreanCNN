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


def center_image(image, target_size=None):
    """
    Центрирует содержимое изображения, находя bounding box непустой области
    
    Args:
        image: PIL.Image - входное изображение
        target_size (tuple, optional): Целевой размер (width, height). Если None, сохраняется исходный размер
        
    Returns:
        PIL.Image: Центрированное изображение
    """
    import numpy as np
    from PIL import Image
    
    # Конвертируем в grayscale если нужно
    if image.mode != 'L':
        img_gray = image.convert('L')
    else:
        img_gray = image
    
    # Преобразуем в numpy array
    img_array = np.array(img_gray)
    
    # Находим bounding box непустой области
    # Ищем все пиксели, которые не являются фоном
    # Используем адаптивный порог: находим разницу между максимумом и минимумом
    img_min = img_array.min()
    img_max = img_array.max()
    
    if img_max > img_min:
        # Используем адаптивный порог: считаем фоном пиксели близкие к максимуму
        # Для темных символов на светлом фоне используем порог на 15% ниже максимума
        # Это позволяет лучше находить темные символы даже если они не очень темные
        threshold = img_min + (img_max - img_min) * 0.85
        coords = np.where(img_array < threshold)
    else:
        # Если изображение однотонное, возвращаем как есть
        coords = (np.array([]), np.array([]))
    
    if len(coords[0]) > 0:
        # Находим границы
        top = coords[0].min()
        bottom = coords[0].max() + 1
        left = coords[1].min()
        right = coords[1].max() + 1
        
        # Определяем размер исходного изображения
        img_height, img_width = img_array.shape
        
        # Вычисляем центр bounding box
        bbox_center_x = (left + right) / 2
        bbox_center_y = (top + bottom) / 2
        
        # Вычисляем центр изображения
        img_center_x = img_width / 2
        img_center_y = img_height / 2
        
        # Вычисляем смещение от центра (в процентах от размера изображения)
        offset_x_percent = abs(bbox_center_x - img_center_x) / img_width
        offset_y_percent = abs(bbox_center_y - img_center_y) / img_height
        
        # Порог: если смещение меньше 15% от размера, считаем что уже по центру
        # Увеличенный порог для рукописных изображений, которые могут быть немного смещены
        # но при этом уже достаточно центрированы и не требуют перецентрирования
        CENTER_THRESHOLD = 0.15
        
        # Если содержимое уже примерно по центру, просто изменяем размер без центрирования
        if offset_x_percent < CENTER_THRESHOLD and offset_y_percent < CENTER_THRESHOLD:
            if target_size:
                return img_gray.resize(target_size, Image.Resampling.LANCZOS)
            return img_gray
        
        # Если не по центру, центрируем
        # Вырезаем область с содержимым
        content_region = img_array[top:bottom, left:right]
        
        # Определяем размер выходного изображения
        if target_size:
            out_width, out_height = target_size
        else:
            out_width, out_height = image.size
        
        # Создаем новое изображение с фоном (максимальное значение из исходного)
        bg_color = int(img_array.max())
        centered_img = Image.new('L', (out_width, out_height), color=bg_color)
        
        # Вычисляем позицию для центрирования
        content_height, content_width = content_region.shape
        paste_x = (out_width - content_width) // 2
        paste_y = (out_height - content_height) // 2
        
        # Вставляем содержимое в центр
        content_pil = Image.fromarray(content_region)
        centered_img.paste(content_pil, (paste_x, paste_y))
        
        return centered_img
    else:
        # Если не нашли содержимое, возвращаем исходное изображение с измененным размером если нужно
        if target_size:
            return img_gray.resize(target_size, Image.Resampling.LANCZOS)
        return img_gray


class ImageTransform:
    """
    Класс для трансформаций изображений без использования torchvision
    """
    def __init__(self, resize_size, normalize=True, center_image_flag=True):
        """
        Args:
            resize_size (tuple): Размер для изменения размера (height, width)
            normalize (bool): Нормализовать ли изображение в [-1, 1]
            center_image_flag (bool): Центрировать ли изображение перед resize
        """
        self.resize_size = resize_size
        self.normalize = normalize
        self.center_image_flag = center_image_flag
    
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
            
            # Центрируем изображение перед изменением размера (если включено)
            if self.center_image_flag:
                image = center_image(image, target_size=self.resize_size[::-1])  # PIL использует (width, height)
            else:
                # Изменяем размер без центрирования
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
        is_training (bool): Если True, добавляются аугментации для обучения (в текущей реализации не используются)
        
    Returns:
        ImageTransform: Объект трансформации
    """
    return ImageTransform(resize_size=IMAGE_SIZE, normalize=True, center_image_flag=True)


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
