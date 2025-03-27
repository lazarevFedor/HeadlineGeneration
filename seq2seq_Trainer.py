import torch
import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (GPT2Tokenizer, GPT2LMHeadModel,
                          DataCollatorForLanguageModeling,
                          TrainingArguments, Trainer)
import torch.distributed as dist
import torch.multiprocessing as mp

# Функция для очистки текста
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^а-яА-ЯёЁ .,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Дополнительная фильтрация текста
def has_letters(text):
    return bool(re.search(r"[а-яА-ЯёЁ]", text))

# Токенизация
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

file_path = '/kaggle/input/corpus-of-russian-news-articles-from-lenta/lenta-ru-news.csv'

df = pd.read_csv(file_path, low_memory=False)

# Оставляем только нужные колонки
df = df[['title', 'text']]

# Очистка текста
df['text'] = df['text'].apply(clean_text)
df['title'] = df['title'].apply(clean_text)

# Убираем строки без букв
df = df[df['text'].apply(has_letters) & df['title'].apply(has_letters)]
# Сбрасываем индексы
df = df.reset_index(drop=True)

# Подготовка данных для обучения
df['input'] = "Текст новости: " + df['text'] + " [SEP] Заголовок:"
df['target'] = df['title']
df = df.sample(5000, random_state=42)

# Разделение на тренировочный и валидационный наборы
train_data, val_data = train_test_split(df, test_size=0.1, random_state=42)

# Преобразование в Dataset
train_dataset = Dataset.from_dict({"text": train_data['input'].tolist(),
                                   "label": train_data['target'].tolist()})
val_dataset = Dataset.from_dict({"text": val_data['input'].tolist(),
                                 "label": val_data['target'].tolist()})

# Подключение токенизатора
tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Применение токенизация
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Удаляем ненужные колонки
train_dataset = train_dataset.remove_columns(["text", "label"])
val_dataset = val_dataset.remove_columns(["text", "label"])

# Преобразование в PyTorch
train_dataset.set_format("torch")
val_dataset.set_format("torch")

# Функция для запуска DDP

def train_ddp(rank, world_size):
    print(f"[Process {rank}] Инициализация процесса...")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"[Process {rank}] Инициализация завершена")

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    print(f"[Process {rank}] Устройство установлено: {device}")

    model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2").to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    print(f"[Process {rank}] Модель загружена")

    tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"[Process {rank}] Токенизатор загружен")

    # Добавляем семплеры для каждого процесса
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, sampler=val_sampler)
    print(f"[Process {rank}] Датасеты загружены")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        eval_steps=500,
        logging_strategy="steps",
        logging_steps=500,
        save_strategy="steps",
        save_steps=5000,
        num_train_epochs=1,  # Уменьшим для теста
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        report_to="none",
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    try:
        print(f"[Process {rank}] Начало обучения...")
        trainer.train()
        print(f"[Process {rank}] Обучение завершено успешно")
    except Exception as e:
        print(f"[Process {rank}] Ошибка во время обучения: {e}")
        dist.destroy_process_group()
        raise e

    # Сохраняем только на процессе rank=0
    if rank == 0:
        print(f"[Process {rank}] Сохранение модели...")
        model.module.save_pretrained("./model")
        tokenizer.save_pretrained("./model")
        print(f"[Process {rank}] Модель сохранена")

    # Завершаем процесс
    dist.destroy_process_group()
    print(f"[Process {rank}] Процесс завершён успешно")

# Основной запуск с несколькими процессами
def main():
    world_size = torch.cuda.device_count()
    print(f"Используется {world_size} GPU для обучения.")
    
    mp.spawn(train_ddp,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()

# Функция генерации заголовков
def generate_title(news_text, model, tokenizer, device):
    input_text = f"Текст новости: {news_text} [SEP] Заголовок:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    output = model.generate(
        input_ids,
        min_length=5,
        max_new_tokens=20,
        num_beams=10,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        do_sample=True,
        early_stopping=True
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_title = generated_text.split("Заголовок:")[-1].strip()
    return generated_title

if torch.cuda.is_available():
    model = GPT2LMHeadModel.from_pretrained("./model").cuda()
    tokenizer = GPT2Tokenizer.from_pretrained("./model")

    article = (
        "Россия в ближайшие годы будет наращивать объем финансирования исследований в сфере искусственного интеллекта, "
        "заявил вице-премьер РФ Дмитрий Чернышенко. Искусственный интеллект – это та прорывная и быстрая технология, "
        "которая важна как для гражданских, так и для военных нужд."
    )

    model.eval()
    generated_title = generate_title(article, model, tokenizer, device='cuda')
    print("Текст новости:", article)
    print("Сгенерированный заголовок:", generated_title)
!zip -r models.zip ./model
