import torch
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (GPT2Tokenizer, GPT2LMHeadModel,
                          DataCollatorForLanguageModeling,
                          TrainingArguments, Trainer)

# Функция для очистки текста
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^а-яА-ЯёЁ .,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Дополнительная фильтрация: убираем строки, где нет букв (например, содержатся только пробелы или пунктуация)
def has_letters(text):
    # Проверяем наличие хотя бы одной русской буквы в тексте
    return bool(re.search(r"[а-яА-ЯёЁ]", text))


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)


file_path = "/kaggle/input/corpus-of-russian-news-articles-from-lenta/lenta-ru-news.csv"

df = pd.read_csv(file_path, low_memory=False)


# Оставляем только нужные колонки: 'title' и 'text'
df = df[['title', 'text']]

# Очистка текста
df['text'] = df['text'].apply(clean_text)
df['title'] = df['title'].apply(clean_text)

#избавляемся от ненужной пунктуации
df = df[df['text'].apply(has_letters) & df['title'].apply(has_letters)]

# Сбрасываем индексы
df = df.reset_index(drop=True)


# Подготовка входных данных и целевых значений для обучения
df['input'] = "Текст новости: " + df['text'] + " [SEP] Заголовок:"
df['target'] = df['title']
df = df.sample(50000, random_state=42)

# Разделение датасета на выборки
train_data, val_data = train_test_split(df, test_size=0.1, random_state=42)

# Преобразование в формат Dataset
train_dataset = Dataset.from_dict({"text": train_data['input'].tolist(),
                                   "label": train_data['target'].tolist()})
val_dataset = Dataset.from_dict({"text": val_data['input'].tolist(),
                                 "label": val_data['target'].tolist()})

print("Пример обучающего текста из набора:", train_dataset[0])

# Токенизация
tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")

# Устанавливаем токен паддинга равным токену конца текста
tokenizer.pad_token = tokenizer.eos_token

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Оставляем только нужные колонки
train_dataset = train_dataset.remove_columns(["text", "label"])
val_dataset = val_dataset.remove_columns(["text", "label"])

# Преобразование в PyTorch
train_dataset.set_format("torch")
val_dataset.set_format("torch")


# Загрузка модели и создание Data collator
model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=5000,
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="steps",
    save_steps=5000,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    report_to="none",
    fp16=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    data_collator=data_collator
)

trainer.train()

# Определение функции генерации заголовков

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def generate_title(news_text, model, tokenizer):
    input_text = f"Текст новости: {news_text} [SEP] Заголовок:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    output = model.generate(
        input_ids,
        min_length=5,  # Минимальная длина заголовка
        max_new_tokens=20,  # Максимальная длина заголовка
        num_beams=10,  # Beam Search для улучшения качества
        no_repeat_ngram_size=2,  # Предотвращение повторений
        temperature=0.7,  # Контроль случайности
        top_k=50,  # Убираем маловероятные токены
        top_p=0.9,  # Top-p sampling
        do_sample=True,  # Включаем сэмплирование
        early_stopping=True  # Прерываем генерацию при достижении токена конца
    )

    # Декодирование и возврат заголовка
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_title = generated_text.split("Заголовок:")[-1].strip()
    return generated_title

article = (
    "Россия в ближайшие годы будет наращивать объем финансирования исследований в сфере искусственного интеллекта, "
    "заявил вице-премьер РФ Дмитрий Чернышенко. Искусственный интеллект – это та прорывная и быстрая технология, "
    "которая важна как для гражданских, так и для военных нужд."
)


generated_title = generate_title(article, model, tokenizer)
print("Текст новости:", article)
print("Сгенерированный заголовок:", generated_title)

output_dir = "."
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

