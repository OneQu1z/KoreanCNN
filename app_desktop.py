"""
Desktop приложение для распознавания корейских букв
Создано с использованием CustomTkinter
"""
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import torch
import threading
import traceback
import queue

from src.config import MODELS_DIR, CLASS_LABELS
from src.predict import load_model, preprocess_image, predict


# Настройка темы CustomTkinter
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class KoreanLetterRecognitionApp(ctk.CTk):
    """Главное окно приложения для распознавания корейских букв"""
    
    def __init__(self):
        super().__init__()
        
        self.title("Распознавание корейских букв")
        self.geometry("1000x650")
        self.resizable(True, True)
        
        # Переменные
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_image_path = None
        self.current_image = None
        self.model_status_label = None  # Будет создан в create_widgets
        self.result_queue = queue.Queue()  # Очередь для передачи результатов из потока
        
        # Создаем интерфейс
        self.create_widgets()
        
        # Загружаем модель после создания интерфейса
        self.load_model_automatically()
    
    def load_model_automatically(self):
        """Автоматически загружает последнюю модель при запуске"""
        try:
            model_files = list(MODELS_DIR.glob("*.pth"))
            if not model_files:
                if self.model_status_label:
                    self.model_status_label.configure(
                        text="❌ Модели не найдены в папке models/",
                        text_color="red"
                    )
                return
            
            # Сортируем по времени модификации (последняя - самая новая)
            model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            model_path = model_files[0]
            
            self.model, _ = load_model(model_path, self.device)
            if self.model_status_label:
                self.model_status_label.configure(
                    text=f"✅ Модель загружена: {model_path.name}",
                    text_color="green"
                )
        except Exception as e:
            if self.model_status_label:
                self.model_status_label.configure(
                    text=f"❌ Ошибка загрузки модели: {str(e)}",
                    text_color="red"
                )
    
    def create_widgets(self):
        """Создает виджеты интерфейса"""
        
        # Заголовок
        title_label = ctk.CTkLabel(
            self,
            text="🇰🇷 Распознавание корейских букв",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=20)
        
        # Статус модели
        self.model_status_label = ctk.CTkLabel(
            self,
            text="Загрузка модели...",
            font=ctk.CTkFont(size=12)
        )
        self.model_status_label.pack(pady=5)
        
        # Разделитель
        separator1 = ctk.CTkFrame(self, height=2, fg_color="gray")
        separator1.pack(fill="x", padx=20, pady=10)
        
        # Основной контейнер с двумя колонками
        main_container = ctk.CTkFrame(self)
        main_container.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Левая колонка - загрузка и изображение
        left_frame = ctk.CTkFrame(main_container)
        left_frame.pack(side="left", fill="both", expand=False, padx=10, pady=10)
        left_frame.configure(width=350)
        
        file_label = ctk.CTkLabel(
            left_frame,
            text="📤 Загрузка изображения",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        file_label.pack(pady=10)
        
        # Кнопка выбора файла
        self.select_file_btn = ctk.CTkButton(
            left_frame,
            text="Выбрать файл",
            command=self.select_file,
            font=ctk.CTkFont(size=14),
            width=200,
            height=40
        )
        self.select_file_btn.pack(pady=10)
        
        # Путь к файлу
        self.file_path_label = ctk.CTkLabel(
            left_frame,
            text="Файл не выбран",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            wraplength=300
        )
        self.file_path_label.pack(pady=5)
        
        # Превью изображения (компактнее)
        self.image_preview_label = ctk.CTkLabel(
            left_frame,
            text="",
            width=250,
            height=250
        )
        self.image_preview_label.pack(pady=10)
        
        # Кнопка распознавания
        self.recognize_btn = ctk.CTkButton(
            left_frame,
            text="🔎 Распознать",
            command=self.recognize_image,
            font=ctk.CTkFont(size=15, weight="bold"),
            width=200,
            height=45,
            state="disabled"
        )
        self.recognize_btn.pack(pady=15)
        
        # Правая колонка - результаты
        right_frame = ctk.CTkFrame(main_container)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        results_label = ctk.CTkLabel(
            right_frame,
            text="📊 Результаты распознавания",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        results_label.pack(pady=10)
        
        # Основной результат (крупнее и заметнее)
        result_container = ctk.CTkFrame(right_frame)
        result_container.pack(pady=15, padx=20, fill="x")
        
        self.result_label = ctk.CTkLabel(
            result_container,
            text="—",
            font=ctk.CTkFont(size=48, weight="bold"),
            text_color="gray"
        )
        self.result_label.pack(pady=15)
        
        # Уверенность
        self.confidence_label = ctk.CTkLabel(
            result_container,
            text="",
            font=ctk.CTkFont(size=16),
            text_color="gray"
        )
        self.confidence_label.pack(pady=5)
        
        # Вероятности по классам (с прокруткой)
        prob_label = ctk.CTkLabel(
            right_frame,
            text="Вероятности по классам:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        prob_label.pack(pady=(20, 5), padx=20, anchor="w")
        
        # Scrollable frame для вероятностей
        self.probabilities_scroll = ctk.CTkScrollableFrame(
            right_frame,
            height=250
        )
        self.probabilities_scroll.pack(fill="both", expand=True, padx=20, pady=5)
        
        # Используем сам scrollable frame как контейнер для вероятностей
        self.probabilities_frame = self.probabilities_scroll
        
        # Информация о классах (внизу)
        info_text = "Распознаваемые буквы: " + ", ".join(CLASS_LABELS)
        info_label = ctk.CTkLabel(
            self,
            text=info_text,
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        info_label.pack(pady=5)
    
    def select_file(self):
        """Открывает диалог выбора файла"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[
                ("Изображения", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("Все файлы", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = Path(file_path)
            self.file_path_label.configure(
                text=f"Файл: {self.current_image_path.name}",
                text_color="black"
            )
            
            # Загружаем и показываем изображение
            try:
                image = Image.open(file_path)
                
                # Сохраняем оригинальное изображение
                self.current_image = image.copy()
                
                # Изменяем размер для превью (максимум 250x250 для компактности)
                max_size = 250
                preview_image = image.copy()
                preview_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Конвертируем в CTkImage для отображения (правильная работа с HighDPI)
                ctk_image = ctk.CTkImage(light_image=preview_image, dark_image=preview_image, size=preview_image.size)
                self.image_preview_label.configure(image=ctk_image, text="")
                self.image_preview_label.image = ctk_image  # Сохраняем ссылку
                
                # Активируем кнопку распознавания
                self.recognize_btn.configure(state="normal")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить изображение:\n{str(e)}")
                self.recognize_btn.configure(state="disabled")
    
    def recognize_image(self):
        """Распознает изображение"""
        if self.model is None:
            messagebox.showerror("Ошибка", "Модель не загружена!")
            return
        
        if self.current_image is None:
            messagebox.showerror("Ошибка", "Сначала выберите изображение!")
            return
        
        # Отключаем кнопку во время обработки
        self.recognize_btn.configure(state="disabled", text="Обработка...")
        self.update()  # Принудительно обновляем UI
        
        # Очищаем очередь результатов
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        
        # Запускаем распознавание в отдельном потоке
        thread = threading.Thread(target=self._recognize_thread, name="RecognitionThread")
        thread.daemon = True
        thread.start()
        
        # Запускаем проверку результатов
        self.check_results()
    
    def _recognize_thread(self):
        """Поток для распознавания (чтобы не блокировать UI)"""
        try:
            # Сохраняем временный файл
            temp_path = Path("temp_recognition.png")
            if self.current_image is None:
                raise ValueError("Изображение не загружено")
            
            self.current_image.save(temp_path)
            
            # Предобрабатываем изображение
            image_tensor = preprocess_image(temp_path)
            
            # Делаем предсказание
            predicted_class, confidence, all_probs = predict(
                self.model,
                image_tensor,
                self.device
            )
            
            # Удаляем временный файл
            if temp_path.exists():
                temp_path.unlink()
            
            # Отправляем результат в очередь
            self.result_queue.put(("success", predicted_class, confidence, all_probs))
            
        except Exception as e:
            error_msg = f"Ошибка при распознавании:\n{str(e)}\n\nДетали:\n{traceback.format_exc()}"
            # Отправляем ошибку в очередь
            self.result_queue.put(("error", error_msg))
    
    def check_results(self):
        """Проверяет очередь результатов и обновляет UI"""
        try:
            # Проверяем очередь без блокировки
            result = self.result_queue.get_nowait()
            
            if result[0] == "success":
                _, predicted_class, confidence, all_probs = result
                self._update_results(predicted_class, confidence, all_probs)
                self.recognize_btn.configure(state="normal", text="🔎 Распознать")
            elif result[0] == "error":
                _, error_msg = result
                messagebox.showerror("Ошибка", error_msg)
                self.recognize_btn.configure(state="normal", text="🔎 Распознать")
        except queue.Empty:
            # Очередь пуста, проверяем снова через 100мс
            self.after(100, self.check_results)
    
    def _update_results(self, predicted_class, confidence, all_probs):
        """Обновляет результаты распознавания в UI"""
        # Основной результат
        self.result_label.configure(
            text=predicted_class,
            text_color="green"
        )
        
        # Уверенность
        self.confidence_label.configure(
            text=f"Уверенность: {confidence * 100:.2f}%",
            text_color="black"
        )
        
        # Очищаем старые вероятности
        for widget in self.probabilities_frame.winfo_children():
            widget.destroy()
        
        # Создаем виджеты для вероятностей (более компактные)
        for label, prob in zip(CLASS_LABELS, all_probs):
            prob_frame = ctk.CTkFrame(self.probabilities_frame)
            prob_frame.pack(fill="x", padx=5, pady=3)
            
            # Метка класса
            class_label = ctk.CTkLabel(
                prob_frame,
                text=label,
                font=ctk.CTkFont(size=18, weight="bold"),
                width=40
            )
            class_label.pack(side="left", padx=8)
            
            # Прогресс-бар
            progress = ctk.CTkProgressBar(prob_frame, height=20)
            progress.pack(side="left", fill="x", expand=True, padx=8)
            progress.set(prob)
            
            # Процент
            percent_label = ctk.CTkLabel(
                prob_frame,
                text=f"{prob * 100:.1f}%",
                font=ctk.CTkFont(size=13),
                width=55
            )
            percent_label.pack(side="left", padx=8)
            
            # Выделяем предсказанный класс
            if label == predicted_class:
                prob_frame.configure(fg_color="#90EE90")  # Светло-зеленый
                class_label.configure(text_color="darkgreen", font=ctk.CTkFont(size=20, weight="bold"))
                percent_label.configure(font=ctk.CTkFont(size=14, weight="bold"))
        
        # Включаем кнопку обратно
        self.recognize_btn.configure(state="normal", text="🔎 Распознать")


def main():
    """Главная функция для запуска приложения"""
    app = KoreanLetterRecognitionApp()
    app.mainloop()


if __name__ == "__main__":
    main()

