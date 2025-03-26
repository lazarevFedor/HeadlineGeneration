import numpy as np
import pandas as pd
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Attention
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences


# Чтение данных
data_path = 'news_summary.csv'
df = pd.read_csv(data_path, encoding='windows-1251')
articles = df['text'].values
summaries = df['headlines'].values

# токенизация
article_tokenizer = Tokenizer()
article_tokenizer.fit_on_texts(articles)
article_sequences = article_tokenizer.texts_to_sequences(articles)
article_word_index = article_tokenizer.word_index

summary_tokenizer = Tokenizer()
summary_tokenizer.fit_on_texts(summaries)
summary_sequences = summary_tokenizer.texts_to_sequences(summaries)
summary_word_index = summary_tokenizer.word_index

# паддинг последовательностей
max_len_article = 300
max_len_summary = 50

encoder_input_data = pad_sequences(article_sequences, maxlen=max_len_article, padding='post')
decoder_input_data = pad_sequences(summary_sequences, maxlen=max_len_summary, padding='post')

# смещение на один токен вправо для decoder_target_data
decoder_target_data = np.zeros((len(summaries), max_len_summary, len(summary_word_index) + 1), dtype='float32')
for i, seqs in enumerate(summary_sequences):
    for t, token in enumerate(seqs):
        if t > 0:
            decoder_target_data[i, t - 1, token] = 1.0

# гиперпараметры
latent_dim = 256
num_encoder_tokens = len(article_word_index) + 1
num_decoder_tokens = len(summary_word_index) + 1

# энкодер
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_dim=num_encoder_tokens, output_dim=latent_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# декодер
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(input_dim=num_decoder_tokens, output_dim=latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

# механизм внимания
attention = Attention()([decoder_outputs, encoder_outputs])
decoder_combined_context = Concatenate(axis=-1)([decoder_outputs, attention])

# выходной слой
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_combined_context)

# модель Seq2Seq
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=64, epochs=1, validation_split=0.2)
model.save('my_model.keras')
model.save('seq2seq_model.h5')