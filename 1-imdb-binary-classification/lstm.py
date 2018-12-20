from keras import layers, models
from keras.datasets import imdb
from keras.preprocessing import sequence

import matplotlib.pyplot as plt

max_features = 10000
maxlen = 500
batch_size = 32
lstm_width = 32
embedding_dim = 32

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
	num_words=max_features)

train_data = sequence.pad_sequences(train_data, maxlen=maxlen)
test_data = sequence.pad_sequences(test_data, maxlen=maxlen)

model = models.Sequential()
model.add(layers.Embedding(max_features, embedding_dim))
model.add(layers.LSTM(lstm_width))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(train_data, train_labels, 
	epochs = 10,
	batch_size=batch_size,
	validation_split=0.2)

acc = history['acc']
val_acc = history['val_acc']
loss = history['loss']
val_loss = history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', title="Training Accuracy")
plt.plot(epochs, val_acc, 'b', title="Validation Accuracy")

plt.figure()
plt.plot(epochs, loss, 'bo', title="Training Loss")
plt.plot(epochs, val_loss, 'b', title="Validation Loss")