from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical

# Import labeled training and testing data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Set input and training parameters
(TRAIN_N, HEIGHT, WIDTH) = train_images.shape
(TEST_N, _, _) = test_images.shape
BATCH_SIZE = 128
NUM_EPOCHS = 5

# Reshape & retype images and labels
train_images = train_images.reshape((TRAIN_N, WIDTH * HEIGHT))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((TEST_N, WIDTH * HEIGHT))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Build the network with a single hidden layer
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(WIDTH * HEIGHT, )))
network.add(layers.Dense(10, activation='softmax'))

# Compile the network
network.compile(optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# Train the network
network.fit(train_images, 
    train_labels, 
    epochs=NUM_EPOCHS, 
    batch_size=BATCH_SIZE)

# Output training results
test_loss, test_accuracy = network.evaluate(test_images, test_labels)
print(test_loss, test_accuracy)
