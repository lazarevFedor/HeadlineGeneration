import torch
import pandas as pd
import re
from datetime import datetime
from sklearn.model_selection import train_test_split
from datasets import Dataset
import evaluate
from transformers import (GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling,
                          TrainingArguments, Trainer, TrainerCallback)
import shutil




def main():
    DS_AMOUNT = 150000  # кол-во данных для обучения
    OUTPUT_DIR = "./modelFolder"

    class ShowExamplesCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, **kwargs):
            print("\n📊 Примеры генерации на валидации:")
            model.eval()
            for i in range(3):
                input_text = val_data['input'].iloc[i]
                reference = val_data['target'].iloc[i]
                news = input_text.replace("Текст новости: ", "").split(" [SEP]")[0]
                generated = generate_title(news)
                print(f"\n📰 Новости: {news[:250]}...")
                print(f"✅ Оригинал: {reference}")
                print(f"🤖 Модель:   {generated}")
            print("-" * 80)

    # Логгер
    def log(msg):
        print(f"{datetime.now()} : {msg}")

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return rouge.compute(predictions=preds, references=labels)
    # Функция для очистки текста;
    # Работаем только с Кириллицей и знаками препинания;
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r"[^а-яА-ЯёЁ .,!?]", "", text)  # Оставляем только Кириллицу
        text = re.sub(r"\s+", " ", text).strip()  # Заменяем множественные пробелы на 1
        return text

    # Дополнительная фильтрация: наличие хотя бы одной русской буквы в тексте
    def has_letters(text):
        return bool(re.search(r"[а-яА-ЯёЁ]", text))

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    # Определение функции генерации заголовков
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
        # Постобработка
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

    def remove_patterns(text):
        text = re.sub(r"в номер.*", "", text)
        text = re.sub(r"Видео.*", "", text)
        return text.strip()


    file_path = "/kaggle/input/corpus-of-russian-news-articles-from-lenta/lenta-ru-news.csv"

    df = pd.read_csv(file_path, low_memory=False)
    log("Датасет считан")

    # Оставляем только нужные колонки: 'title' и 'text'
    df = df[['title', 'text']]
    log("Из датасета удалена лишняя информация")

    # Очистка текста
    df['text'] = df['text'].apply(clean_text)
    df['title'] = df['title'].apply(clean_text)

    # Избавляемся от ненужной пунктуации
    df = df[df['text'].apply(has_letters) & df['title'].apply(has_letters)]

    #Избавляемся от хвостов "в номере от мая"/"видео доступно"
    df['title'] = df['title'].apply(remove_patterns)

    log("Датасет очищен")

    # Переиндексирование датасета
    df = df.reset_index(drop=True)
    log("Датасет переиндексирован")

    # Подготовка входных данных и целевых значений для обучения

    df['input'] = "Текст новости: " + df['text'] + " [SEP] Заголовок:"
    df['target'] = df['title']
    df = df.sample(DS_AMOUNT, random_state=42)
    log(f"Выбрано {DS_AMOUNT} входных и целевых значений для обучения")

    # Разделение датасета на выборки
    train_data, val_data = train_test_split(df, test_size=0.1, random_state=42)

    # Преобразование в формат Dataset
    train_dataset = Dataset.from_dict({"text": train_data['input'].tolist(),
                                       "label": train_data['target'].tolist()})
    val_dataset = Dataset.from_dict({"text": val_data['input'].tolist(),
                                     "label": val_data['target'].tolist()})

    log("Преобразование в таблички в Dataset object")
    log(f"Пример обучающего текста из набора : {train_dataset[0]}")

    # Токенизация
    tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")

    # Устанавливаем токен паддинга равным токену конца текста
    tokenizer.pad_token = tokenizer.eos_token
    log("Токенизатор инициализирован")

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    log("Датасет токенизирован")

    # Оставляем только нужные колонки
    train_dataset = train_dataset.remove_columns(["text", "label"])
    val_dataset = val_dataset.remove_columns(["text", "label"])
    log("Из датасета убраны лишние данные(столбцы)")

    # Преобразование в PyTorch
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    log("Датасет преобразован в тензоры для подачи в модель")

    # Загрузка модели и создание Data collator
    model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
    log("Предобученная языковая модель инциализирована")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    log("Батчи данных подготовлены в collator")

    training_args = TrainingArguments(
        ddp_find_unused_parameters=False,
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=12,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        report_to="none",
        fp16=True,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics = compute_metrics,
        callbacks = [ShowExamplesCallback()]
    )

    log("Тренер готов к работе")

    trainer.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("🔍 2.Используем DDP:", torch.distributed.is_initialized())
    print("🧠 2.Текущий ранг процесса:", torch.distributed.get_rank() if torch.distributed.is_initialized() else "N/A")


    article = (
        "Россия в ближайшие годы будет наращивать объем финансирования исследований в сфере искусственного интеллекта, "
        "заявил вице-премьер РФ Дмитрий Чернышенко. Искусственный интеллект – это та прорывная и быстрая технология, "
        "которая важна как для гражданских, так и для военных нужд."
    )

    generated_title = generate_title(article, model, tokenizer)
    print("Текст новости:", article)
    print("Сгенерированный заголовок:", generated_title)

    # сохраняет фин версию, но можно и без нее, тк тренер автоматом сохраняет финальную лучшую версию
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Архивация модели в ZIP-архив
    shutil.make_archive("trained_model", 'zip', OUTPUT_DIR)
    log("Финальная модель сохранена и заархивирована как trained_model.zip")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Архивация модели в ZIP-архив
    shutil.make_archive("trained_model", 'zip', OUTPUT_DIR)
    log("Финальная модель сохранена и заархивирована как trained_model.zip")


if __name__ == "__main__":
    main()