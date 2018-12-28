from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras import models, layers

import numpy as np

import os

# Import texts and labels from data
base_dir = os.getcwd()
data_dir = os.path.join(base_dir, 'data')
train_dir = os.path.join(data_dir, 'train')

labels = []
texts = []

for label in ['pos', 'neg']:
	dir_name = os.path.join(train_dir, label)
	for fname in os.listdir(dir_name):
		if fname[-4:] == '.txt':
			file = open(os.path.join(dir_name, fname))
			content = file.read()
			texts.append(content)
			file.close()
			if (label == 'pos'):
				labels.append(0)
			else:
				labels.append(1)

# Vectorize inputs and outputs
maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000
embedding_dim = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print("Found %s unique tokens" % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)

print("Shape of data tensor: ", data.shape)
print("Shape of label tensor: ", labels.shape)

# Shuffle the data and labels
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples : training_samples + validation_samples]
y_val = labels[training_samples : training_samples + validation_samples]

# Build GloVe index
glove_dir = os.path.join(data_dir, 'glove')
glove_file = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
glove_index = {}
for line in glove_file:
	values = line.split()
	word = values[0]
	coefficients = np.asarray(values[1:], dtype='float32')
	glove_index[word] = coefficients
glove_file.close()

print('Imported GloVe index with %s words' % len(glove_index))

# Build embedding matrix
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
	embedding = glove_index.get(word)
	if (i < max_words and embedding is not None):
		embedding_matrix[i] = embedding

# Compile model, set and freeze embedding layer weights
model = models.Sequential()
model.add(layers.Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, 
	epochs=10, 
	batch_size=32, 
	validation_data=(x_val, y_val))

model.save_weights('pre_trained_glove_model.h5')
