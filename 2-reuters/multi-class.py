from keras import models, layers, losses, optimizers, metrics
from keras.datasets import reuters
import matplotlib.pyplot as plt

# Restrict dataset to NUM_WORDS most frequent words
NUM_WORDS = 10000

NUM_EPOCHS = 10
BATCH_SIZE = 512

# Load labeled training and test data
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=NUM_WORDS)

# Takes in a list of word sequences and outputs a list of one-hot encoded words
def vectorize_sequences(sequences, dimension=NUM_WORDS):
	results = zeros(len(sequences), dimension)
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1
	return results

# Alternatively use keras.utils.np_tools.to_categorizal
one_hot_train_labels = vectorize_sequences(train_labels)
one_hot_test_labels = vectorize_sequences(test_labels)

# Build and compile a 3 layer model
model = models.Sequential()
model = model.add(layers.Dense(64, activation='relu', input_shape=(NUM_WORDS,)))
model = model.add(layers.Dense(64, activation='relu'))
model = model.add(layers.Dense(len(train_labels), activation='softmax'))
model.compile(optimizer=losses.rmsprop, 
	loss=losses.categorical_crossentropy,
	metrics=[metrics.accuracy])

# Set aside first 1000 data points for validation
val_data = train_data[:1000]
val_labels = train_labels[:1000]
partial_train_data = train_data[1000:]
partial_train_labels = train_labels[1000:]

# Train the model for NUM_EPOCHS epochs
history = model.fit(partial_train_data, 
	partial_train_labels, 
	epochs=NUM_EPOCHS, 
	batch_size=BATCH_SIZE, 
	validation_data=(val_data_, val_labels))

# Graph the training statistics 