"""
Вспомогательные утилиты
"""
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Устанавливает seed для воспроизводимости результатов
    
    Args:
        seed (int): Значение seed для генераторов случайных чисел
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Для CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Дополнительные настройки для детерминированности
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def log_info(message, prefix="[INFO]"):
    """
    Простая функция логирования (на основе print)
    
    Args:
        message (str): Сообщение для вывода
        prefix (str): Префикс для сообщения (по умолчанию [INFO])
    """
    print(f"{prefix} {message}")


def log_warning(message):
    """
    Логирование предупреждений
    
    Args:
        message (str): Сообщение-предупреждение
    """
    log_info(message, prefix="[WARNING]")


def log_error(message):
    """
    Логирование ошибок
    
    Args:
        message (str): Сообщение об ошибке
    """
    log_info(message, prefix="[ERROR]")

