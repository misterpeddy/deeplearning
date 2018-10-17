from keras import metrics, models, layers, losses, optimizers
from keras.datasets import imdb
import matplotlib.pyplot as plt
import numpy as np

# Restrict dataset to NUM_WORDS most frequent words
NUM_WORDS = 10000

# Import labeled traning and test data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=NUM_WORDS)

# Given list of word sequences, return list of one-hot encoded reviews 
def vectorize_sequences(sequences, dimension=NUM_WORDS):
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1
  return results

# Vectorize the training and test data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# Set aside validation set
x_val = x_train[:10000]
y_val = y_train[:10000]
partial_x_train = x_train[10000:]
partial_y_train = y_train[10000:]

# Build and compile a 3 layer network
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
             loss=losses.binary_crossentropy,
             metrics=[metrics.binary_accuracy])

# Train the model using train and validation data
history = model.fit(partial_x_train,
                   partial_y_train,
                   epochs=4,
                   batch_size=512,
                   validation_data=(x_val, y_val))

# Capture training statistics
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
binary_accuracy = history_dict['binary_accuracy']
val_binary_accuracy = history_dict['val_binary_accuracy']

epochs = range(1, len(history_dict.get('binary_accuracy')) + 1)

# Plot the training and validation loss
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the training and validation accuracy
plt.plot(epochs, binary_accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_binary_accuracy, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Run the test data set through the model
model.predict(x_test)