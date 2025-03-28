# 📝 Проект по автореферированию текста

## 🚀 Описание проекта
Проект представляет собой набор алгоритмов для автоматического реферирования текста (генерации краткого содержания). В основе проекта лежат три разных подхода к реферированию:

✅ **LexRank** — алгоритм ранжирования предложений на основе сходства между ними  
✅ **TextRank** — алгоритм ранжирования, основанный на графе предложений  
✅ **Seq2Seq** — обучаемая модель на основе трансформеров для генерации рефератов  

Проект включает интерфейсы для использования моделей и алгоритмов через десктопное приложение и предоставляет возможность дообучения модели Seq2Seq.  

---

## 📁 Структура проекта
├── models
├── examples.txt
├── lex_rank.py
├── seq2seq_app.py
├── seq2seq_trainer.py
├── summarizers_app.py
├── text_rank.py
└── utils

### 🔹 **1. models/**
> В папке хранятся файлы, связанные с обученной Seq2Seq моделью:  
- Конфигурация модели  
- Веса модели  
- Токенизатор  

**⚠️ Важно!**  
👉 Для работы с Seq2Seq моделью необходимо скачать файлы модели из Google Drive:  
[📥 Скачать модель](https://drive.google.com/drive/folders/1Dy2ejyETD-4LrsFh-55nSaiafkR0R83x?usp=share_link)  

1. Перейдите по ссылке  
2. Скачайте содержимое папки  
3. Сохраните в папку `./models`  

---

### 🔹 **2. examples.txt**
> Файл с примерами текстов для автореферирования.  
- Каждая строка — это отдельный текст для обработки.  
- Используется для тестирования алгоритмов и модели.  

---

### 🔹 **3. lex_rank.py**
> Реализация алгоритма **LexRank** — метода ранжирования предложений на основе косинусного сходства:  
- Построение матрицы сходства предложений  
- Построение графа предложений  
- Ранжирование на основе алгоритма PageRank  

**Вход:** текстовая строка  
**Выход:** краткое содержание (суммаризация)  

---

### 🔹 **4. text_rank.py**
> Реализация алгоритма **TextRank** — метода ранжирования предложений с использованием графа:  
- Построение графа предложений  
- Применение алгоритма PageRank  
- Выбор наиболее значимых предложений  

**Вход:** текстовая строка  
**Выход:** краткое содержание (суммаризация)  

---

### 🔹 **5. summarizers_app.py**
> Десктопное приложение на **PyQt5** для использования алгоритмов LexRank и TextRank:  
- Окно для ввода текста  
- Выбор алгоритма (LexRank или TextRank)  
- Кнопка для генерации  
- Поле для вывода результата  

**Как запустить:**  
```bash
python summarizers_app.py
