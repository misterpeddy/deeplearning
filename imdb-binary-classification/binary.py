from keras import metrics, models, layers, losses, optimizers
from keras.datasets import imdb
import numpy as np

NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=NUM_WORDS)


def vectorize_sequences(sequences, dimension=NUM_WORDS):
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1
  return results
  
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# Set aside validation set
x_val = x_train[:10000]
y_val = y_train[:10000]
partial_x_train = x_train[10000:]
partial_y_train = y_train[10000:]

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
             loss=losses.binary_crossentropy,
             metrics=[metrics.binary_accuracy])

history = model.fit(partial_x_train,
                   partial_y_train,
                   epochs=4,
                   batch_size=512,
                   validation_data=(x_val, y_val))

import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(history_dict.get('binary_accuracy')) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

import matplotlib.pyplot as plt

history_dict = history.history
binary_accuracy = history_dict['binary_accuracy']
val_binary_accuracy = history_dict['val_binary_accuracy']

epochs = range(1, len(history_dict.get('binary_accuracy')) + 1)

plt.plot(epochs, binary_accuracy, 'bo', label='Training loss')
plt.plot(epochs, val_binary_accuracy, 'b', label='Validation loss')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

model.predict(x_test)

