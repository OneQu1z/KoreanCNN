"""
Модуль с архитектурой CNN модели для распознавания корейских букв
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import NUM_CLASSES, IMAGE_SIZE


class KoreanAlphabetCNN(nn.Module):
    """
    Простая CNN архитектура для распознавания корейских букв (ㄱ, ㄴ, ㄷ, ㄹ, ㅁ)
    
    Архитектура:
    - Conv1: 1 -> 16 каналов, kernel 3x3
    - ReLU активация
    - MaxPool: 2x2
    - Conv2: 16 -> 32 каналов, kernel 3x3
    - ReLU активация
    - MaxPool: 2x2
    - Flatten
    - FC1: -> 128 нейронов
    - ReLU активация
    - FC2: -> NUM_CLASSES (5 классов)
    """
    
    def __init__(self, num_classes=NUM_CLASSES, input_size=IMAGE_SIZE):
        """
        Инициализация модели
        
        Args:
            num_classes (int): Количество классов для классификации
            input_size (tuple): Размер входного изображения (height, width)
        """
        super(KoreanAlphabetCNN, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Первый сверточный слой: 1 канал (grayscale) -> 16 каналов
        # kernel_size=3, padding=1 для сохранения размера
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        
        # Первый pooling слой: уменьшает размер в 2 раза
        # 28x28 -> 14x14
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Второй сверточный слой: 16 -> 32 канала
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Второй pooling слой: уменьшает размер в 2 раза
        # 14x14 -> 7x7
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Вычисляем размер после сверток и pooling
        # После pool1: (28, 28) -> (14, 14)
        # После pool2: (14, 14) -> (7, 7)
        # Финальный размер: 32 канала * 7 * 7 = 1568
        self.flatten_size = 32 * (input_size[0] // 4) * (input_size[1] // 4)
        
        # Полносвязные слои
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout для регуляризации (опционально, можно включить позже)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        """
        Прямой проход через сеть
        
        Args:
            x (torch.Tensor): Входной тензор [batch_size, 1, height, width]
            
        Returns:
            torch.Tensor: Выходной тензор [batch_size, num_classes]
        """
        # Первая свертка + ReLU + Pooling
        x = self.conv1(x)  # [batch, 1, 28, 28] -> [batch, 16, 28, 28]
        x = F.relu(x)
        x = self.pool1(x)  # [batch, 16, 28, 28] -> [batch, 16, 14, 14]
        
        # Вторая свертка + ReLU + Pooling
        x = self.conv2(x)  # [batch, 16, 14, 14] -> [batch, 32, 14, 14]
        x = F.relu(x)
        x = self.pool2(x)  # [batch, 32, 14, 14] -> [batch, 32, 7, 7]
        
        # Flatten: преобразуем в вектор
        x = x.view(x.size(0), -1)  # [batch, 32, 7, 7] -> [batch, 1568]
        
        # Первый полносвязный слой + ReLU
        x = self.fc1(x)  # [batch, 1568] -> [batch, 128]
        x = F.relu(x)
        
        # Второй полносвязный слой (выходной)
        x = self.fc2(x)  # [batch, 128] -> [batch, 5]
        
        return x
    
    def get_num_parameters(self):
        """
        Возвращает количество параметров модели
        
        Returns:
            int: Количество обучаемых параметров
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_classes=NUM_CLASSES, input_size=IMAGE_SIZE):
    """
    Фабричная функция для создания модели
    
    Args:
        num_classes (int): Количество классов
        input_size (tuple): Размер входного изображения
        
    Returns:
        KoreanAlphabetCNN: Экземпляр модели
    """
    return KoreanAlphabetCNN(num_classes=num_classes, input_size=input_size)
