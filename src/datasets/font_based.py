"""
Модуль для генерации датасета корейских букв через системные шрифты
"""
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

from src.config import (
    RAW_DATA_DIR, CLASS_LABELS, CLASS_TO_IDX, 
    NUM_CLASSES, IMAGE_SIZE
)
from src.utils import log_info, log_warning


# Список корейских шрифтов для генерации изображений
KOREAN_FONTS = [
    "BlackAndWhitePicture-Regular",
    "ChosunIlboMyeongjo",
    "GamjaFlower-Regular",
    "GothicA1-Black",
    "GothicA1-Bold",
    "GothicA1-ExtraBold",
    "GothicA1-ExtraLight",
    "GothicA1-Light",
    "GothicA1-Medium",
    "GothicA1-Regular",
    "GothicA1-SemiBold",
    "GothicA1-Thin",
    "HiMelody-Regular",
    "IropkeBatangM",
    "NanumBarunGothic",
    "NanumBarunGothic-UltraLight",
    "NanumBarunPen-Regular",
    "NanumBrushScript-Regular",
    "NanumGothic-Bold",
    "NanumGothicCoding-Bold",
    "NanumGothicCoding-Regular",
    "NanumGothic-ExtraBold",
    "NanumGothic-Regular",
    "NanumMyeongjo-Bold",
    "NanumMyeongjo-ExtraBold",
    "NanumMyeongjo-Regular",
    "NanumPenScript-Regular",
    "PoorStory-Regular",
    "SeoulHangangB",
    "SeoulHangangBL",
    "SeoulHangangEB",
    "SeoulHangangL",
    "SeoulHangangM",
    "SourceHanSerifK-Regular",
    "Stylish-Regular",
]


def find_font_path(font_name):
    """
    Ищет путь к шрифту в стандартных директориях Windows
    
    Args:
        font_name (str): Имя шрифта
        
    Returns:
        str or None: Путь к файлу шрифта или None если не найден
    """
    # Стандартные расширения для шрифтов
    extensions = ['.ttf', '.otf', '.ttc']
    
    # Стандартные директории для шрифтов Windows
    font_dirs = [
        "C:/Windows/Fonts",
        os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts"),
    ]
    
    for font_dir in font_dirs:
        if not os.path.exists(font_dir):
            continue
        
        for ext in extensions:
            font_path = os.path.join(font_dir, f"{font_name}{ext}")
            if os.path.exists(font_path):
                return font_path
            
            # Также пробуем варианты с пробелами и без дефисов
            font_path_alt = os.path.join(font_dir, f"{font_name.replace('-', '')}{ext}")
            if os.path.exists(font_path_alt):
                return font_path_alt
    
    return None


def generate_korean_char_image(char, size=(28, 28), font_size=24, font_name=None):
    """
    Генерирует изображение корейской буквы используя указанный корейский шрифт
    
    Args:
        char (str): Корейская буква для генерации (ㄱ, ㄴ, ㄷ, ㄹ, ㅁ)
        size (tuple): Размер изображения (width, height)
        font_size (int): Размер шрифта
        font_name (str, optional): Имя шрифта из KOREAN_FONTS. Если None, выбирается случайный
        
    Returns:
        PIL.Image: Изображение с буквой в grayscale
    """
    # Создаем белое изображение
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Выбираем шрифт
    if font_name is None:
        font_name = random.choice(KOREAN_FONTS)
    
    # Ищем путь к шрифту
    font_path = find_font_path(font_name)
    
    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            # Если не удалось загрузить, используем шрифт по умолчанию
            font = ImageFont.load_default()
            log_warning(f"Не удалось загрузить шрифт {font_name}, используется шрифт по умолчанию")
    else:
        # Если шрифт не найден, используем шрифт по умолчанию
        font = ImageFont.load_default()
        log_warning(f"Шрифт {font_name} не найден, используется шрифт по умолчанию")
    
    # Получаем bounding box текста для правильного центрирования
    # textbbox возвращает (left, top, right, bottom) относительно точки (0, 0)
    bbox = draw.textbbox((0, 0), char, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Пробуем использовать anchor='mm' для центрирования (доступно в PIL 8.0.0+)
    # anchor='mm' означает center-middle (центр по горизонтали и вертикали)
    try:
        center_x = size[0] // 2
        center_y = size[1] // 2
        draw.text((center_x, center_y), char, fill='black', font=font, anchor="mm")
    except (TypeError, ValueError):
        # Если anchor не поддерживается, используем расчет вручную
        # Для draw.text() (x, y) - это позиция baseline (нижняя линия текста)
        
        # Горизонтальное центрирование: центрируем bbox
        target_left = (size[0] - text_width) // 2
        x = target_left - bbox[0]  # Смещаем x на разницу между текущим left и target
        
        # Вертикальное центрирование: центрируем bbox по вертикали
        # bbox[1] - расстояние от baseline (y=0) до top символа
        # bbox[3] - расстояние от baseline (y=0) до bottom символа
        # text_height = bbox[3] - bbox[1]
        # Хотим, чтобы центр bbox был в центре изображения
        bbox_center_y = (bbox[1] + bbox[3]) / 2  # Центр bbox относительно baseline=0
        target_center_y = size[1] / 2  # Центр изображения
        # Нужно найти y так, чтобы bbox_center_y + y = target_center_y
        y = target_center_y - bbox_center_y
        
        draw.text((x, y), char, fill='black', font=font)
    
    # Преобразуем в grayscale
    img = img.convert('L')
    
    return img


def generate_dataset_images(num_samples_per_class=200, output_dir=None, use_all_fonts=True):
    """
    Генерирует изображения корейских букв используя корейские шрифты
    
    Args:
        num_samples_per_class (int): Количество изображений для каждого класса
        output_dir (Path): Директория для сохранения изображений
        use_all_fonts (bool): Использовать ли все доступные шрифты для разнообразия
        
    Returns:
        bool: True если успешно созданы изображения
    """
    output_dir = output_dir or RAW_DATA_DIR
    
    # Создаем директории для каждого класса
    for char in CLASS_LABELS:
        class_dir = output_dir / char
        class_dir.mkdir(parents=True, exist_ok=True)
    
    # Проверяем доступные шрифты
    available_fonts = []
    for font_name in KOREAN_FONTS:
        if find_font_path(font_name):
            available_fonts.append(font_name)
    
    if len(available_fonts) == 0:
        log_warning("Не найдено ни одного корейского шрифта. Используется шрифт по умолчанию.")
        available_fonts = [None]  # Используем шрифт по умолчанию
    
    log_info(f"Найдено {len(available_fonts)} доступных корейских шрифтов")
    log_info(f"Генерация {num_samples_per_class} изображений для каждого из {NUM_CLASSES} классов...")
    
    # Генерируем изображения с разнообразием шрифтов и размеров
    for class_idx, char in enumerate(CLASS_LABELS):
        class_dir = output_dir / char
        
        for i in range(num_samples_per_class):
            # Выбираем случайный шрифт для разнообразия
            if use_all_fonts and available_fonts:
                font_name = random.choice(available_fonts)
            else:
                font_name = None
            
            # Добавляем небольшие вариации размера шрифта
            font_size = random.randint(20, 26)
            img = generate_korean_char_image(
                char, 
                size=IMAGE_SIZE, 
                font_size=font_size,
                font_name=font_name
            )
            
            # Сохраняем изображение с информацией о шрифте в имени файла
            if font_name:
                img_path = class_dir / f"{char}_{font_name}_{i:04d}.png"
            else:
                img_path = class_dir / f"{char}_default_{i:04d}.png"
            img.save(img_path)
        
        log_info(f"  Создано {num_samples_per_class} изображений для класса '{char}'")
    
    log_info(f"Всего создано {num_samples_per_class * NUM_CLASSES} изображений")
    return True


def ensure_dataset_exists(output_dir=None):
    """
    Проверяет наличие датасета и генерирует его если нужно
    
    Args:
        output_dir (Path): Директория для сохранения изображений
        
    Returns:
        bool: True если датасет доступен или успешно создан
    """
    output_dir = output_dir or RAW_DATA_DIR
    
    # Проверяем наличие поддиректорий с классами
    has_classes = all((output_dir / char).exists() for char in CLASS_LABELS)
    if has_classes:
        # Проверяем, есть ли файлы в директориях
        has_files = any((output_dir / CLASS_LABELS[0]).iterdir())
        if has_files:
            log_info(f"Датасет на основе шрифтов найден в {output_dir}")
            return True
    
    # Если данных нет, генерируем их
    log_info("Датасет на основе шрифтов не найден. Генерируем изображения...")
    return generate_dataset_images(num_samples_per_class=200, output_dir=output_dir)


class FontBasedDataset(Dataset):
    """
    Датасет для корейских букв на основе генерации через системные шрифты
    
    Генерирует изображения корейских букв используя доступные системные шрифты Windows.
    Это позволяет быстро получить датасет без необходимости загрузки внешних данных.
    """
    
    def __init__(self, root_dir=None, transform=None, num_samples_per_class=200):
        """
        Инициализация датасета
        
        Args:
            root_dir (Path, optional): Корневая директория с данными
            transform (callable, optional): Трансформации для применения к изображениям
            num_samples_per_class (int): Количество изображений для генерации на класс
        """
        self.root_dir = root_dir or RAW_DATA_DIR
        self.transform = transform
        
        # Убеждаемся, что датасет существует
        ensure_dataset_exists(self.root_dir)
        
        # Загружаем пути к изображениям
        self.samples = []
        self.labels = []
        
        for class_idx, char in enumerate(CLASS_LABELS):
            class_dir = self.root_dir / char
            if class_dir.exists():
                image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
                for img_path in class_dir.iterdir():
                    if img_path.suffix.lower() in image_extensions:
                        self.samples.append(img_path)
                        self.labels.append(class_idx)
        
        if len(self.labels) == 0:
            raise ValueError(f"Не найдено изображений в {self.root_dir}")
        
        log_info(f"FontBasedDataset: загружено {len(self.samples)} изображений из {self.root_dir}")
    
    def __len__(self):
        """Возвращает размер датасета"""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Возвращает один сэмпл из датасета
        
        Args:
            idx (int): Индекс сэмпла
            
        Returns:
            tuple: (image, label) где image - тензор изображения, label - индекс класса
        """
        img_path = self.samples[idx]
        image = Image.open(img_path)
        
        # Применяем трансформации
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label
    
    def get_class_name(self, class_idx):
        """Возвращает имя класса по индексу"""
        return CLASS_LABELS[class_idx]

