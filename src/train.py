"""
Основной модуль для обучения модели
"""
import torch
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import MODELS_DIR


def main():
    """
    Главная функция обучения
    """
    print("=" * 50)
    print("Training pipeline initialized")
    print("=" * 50)
    
    # Проверка доступности CUDA
    print("\nПроверка доступности GPU...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ CUDA доступна!")
        print(f"  Устройство: {torch.cuda.get_device_name(0)}")
        print(f"  Версия CUDA: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("✗ CUDA недоступна, будет использован CPU")
    
    print(f"\nИспользуемое устройство: {device}")
    print(f"Директория для сохранения моделей: {MODELS_DIR}")
    
    print("\n" + "=" * 50)
    print("STEP 0 завершен успешно!")
    print("=" * 50)
    
    # Выход с кодом 0 (успешное завершение)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

