from keras import models, layers, losses, optimizers, metrics
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Restrict dataset to NUM_WORDS most frequent words
NUM_WORDS = 10000

NUM_EPOCHS = 10
BATCH_SIZE = 512

# Given list of word sequences, return list of one-hot encoded reviews
def vectorize_sequences(sequences, dimension=NUM_WORDS):
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1
  return results

# Load labeled training and test data
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=NUM_WORDS)

# One-hot encode sentences and labels
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# Build and compile a 3 layer model
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(NUM_WORDS,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss=losses.categorical_crossentropy,
              metrics=[metrics.categorical_accuracy])

# Set aside first 1000 data points for validation
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# Train the model for NUM_EPOCHS epochs
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=NUM_EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(x_val, y_val))

# Graph the training statistics
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title("Training and validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title("Training and validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()
