import re
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton,
                             QTextEdit, QLabel)
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Путь к сохранённой модели
model_path = "./models/new"

# Загружаем модель и токенизатор
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Переносим модель на GPU, если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Функция генерации заголовка
def generate_title(news_text):
    news_text = news_text[:800]  # Ограничим длину до ~800 символов
    input_text = f"Текст новости: {news_text} [SEP] Заголовок:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    output = model.generate(
        input_ids,
        min_length=5,  # Минимальная длина заголовка
        max_new_tokens=30,  # Максимальная длина заголовка
        num_beams=15,  # Beam Search для повышения качества
        no_repeat_ngram_size=2,  # Предотвращение повторений
        do_sample=False,  # Выключаем сэмплирование
        early_stopping=True  # Прерываем генерацию при достижении конца
    )
    #Постобработка
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_title = generated_text.split("Заголовок:")[-1].strip()
    # Удалим лишние фразы и мусор
    generated_title = re.sub(r'[^а-яА-ЯёЁ0-9 .,!?-]', '', generated_title)
    generated_title = re.sub(r'\s+', ' ', generated_title).strip()

    # Если есть неполный конец — отбросим после последней точки
    if '.' in generated_title:
        generated_title = generated_title.rsplit('.', 1)[0] + '.'

    print(generated_title)
    return generated_title

# Основной класс приложения
class TitleGeneratorApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # Устанавливаем заголовок окна и размеры
        self.setWindowTitle("Генератор заголовков новостей")
        self.setGeometry(200, 200, 600, 400)

        # Основной вертикальный лейаут
        layout = QVBoxLayout()

        # Поле ввода текста новости
        self.input_text = QTextEdit(self)
        self.input_text.setPlaceholderText("Введите текст новости здесь...")
        layout.addWidget(self.input_text)

        # Кнопка генерации заголовка
        self.generate_button = QPushButton("Сгенерировать заголовок", self)
        self.generate_button.clicked.connect(self.on_generate)
        layout.addWidget(self.generate_button)

        # Поле вывода результата
        self.output_label = QLabel("Сгенерированный заголовок появится здесь.", self)
        self.output_label.setWordWrap(True)
        layout.addWidget(self.output_label)

        # Применяем лейаут к окну
        self.setLayout(layout)

    # Метод генерации заголовка по нажатию кнопки
    def on_generate(self):
        news_text = self.input_text.toPlainText()
        if news_text.strip():
            try:
                generated_title = generate_title(news_text)
                self.output_label.setText(f"Сгенерированный заголовок:\n\n{generated_title}")
            except Exception as e:
                self.output_label.setText(f"Ошибка генерации: {str(e)}")
        else:
            self.output_label.setText("Введите текст новости!")

# Запуск приложения
def main():
    app = QApplication(sys.argv)
    ex = TitleGeneratorApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
