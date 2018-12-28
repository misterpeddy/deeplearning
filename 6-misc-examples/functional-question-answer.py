from keras import layers, Input
from keras.models import Model
import numpy as np

question_vocab_size = 1000
question_embedding_dim = 32
question_max_len = 10
text_vocab_size = 10000
text_embedding_dim = 64
text_max_len = 100
answer_vocab_size = 500

lstm_width = 32
num_samples = 100

text_input = Input(shape=(None,), dtype='int32', name='text')
text_embedding = layers.Embedding(text_vocab_size, text_embedding_dim)(text_input)
text_encoding = layers.LSTM(lstm_width)(text_embedding)

question_input = Input(shape=(None,), dtype='int32', name='question')
question_embedding = layers.Embedding(question_vocab_size, question_embedding_dim)(question_input)
question_encoding = layers.LSTM(lstm_width)(question_embedding)

concatenated = layers.concatenate([question_encoding, text_encoding], axis=-1)

answer = layers.Dense(answer_vocab_size, activation='softmax')(concatenated)

model = Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop',
	loss='categorical_crossentropy',
	metrics=['acc'])

questions = np.random.randint(1, question_vocab_size, size=(num_samples, question_max_len))
texts = np.random.randint(1, text_vocab_size, size=(num_samples, text_max_len))
answers = np.random.randint(0, 2, size=(num_samples, answer_vocab_size)) #one-hot encoded

model.fit({'text': texts, 'question': questions}, answers, epochs=10, batch_size=128)