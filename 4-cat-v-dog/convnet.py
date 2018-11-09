from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt

import os

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
			  loss='binary_crossentropy',
			  metrics=['acc'])


train_dir = os.path.join(os.getcwd(), 'train')
validation_dir = os.path.join(os.getcwd(), 'validation')

# Data generators (with augmentation for training)
train_datagen = ImageDataGenerator(
	rescale=1./255,
	rotation_range=40,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
	train_dir,
	target_size=(150, 150),
	batch_size=20,
	class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
	validation_dir,
	target_size=(150, 150),
	batch_size=20,
	class_mode='binary')

history = model.fit_generator(
	train_generator,
	steps_per_epoch=100,
	epochs=30,
	validation_data=validation_generator,
	validation_steps=50)

def plot_loss_and_acc_curves(history):
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs = range(1, len(acc) + 1)

	plt.plot(epochs, acc, 'bo', label='Training Accuracy')
	plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
	plt.title('Training and Validation Accuracy')
	plt.legend()

	plt.figure()

	plt.plot(epochs, loss, 'bo', label='Training Loss')
	plt.plot(epochs, val_loss, 'b', label='Validation Loss')
	plt.title('Training and Validation Loss')
	plt.legend()

	plt.show()

plot_loss_and_acc_curves(history)

model.save('cats_and_dogs_small_2.h5')

