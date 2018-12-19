import ssl
context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)

import os
import numpy as np

from keras import models, layers, optimizers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

# Import VGG16 base, set up dir structure and init DataGen
conv_base = VGG16(weights='imagenet',
				  include_top=False,
				  input_shape=(150, 150, 3))

base_dir = os.join(os.getcwd(), 'data')

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 10

# Returns VGG16 feature vectores and labels for sample_count images in directory
def extract_features(directory, sample_count):
	features = np.zeros(shape=(sample_count, 4, 4, 512))
	labels = np.zeros(shape=(sample_count))
	generator = datagen.flow_from_directory(
		directory,
		target_size=(150, 150),
		batch_size=batch_size,
		class_mode='binary')
	i = 0
	for input_batch, labels_batch in generator:
		features_batch = conv_base.predict(input_batch)
		features[i * batch_size : (i+1) * batch_size] = features_batch
		labels[i * batch_size : (i+1) * batch_size] = labels_batch
		i += 1
		if i * batch_size >= sample_count:
			break
	return features, labels

# Obtain and flatten feature/labels for train/validation/test
train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile and train model with feature vectors as input
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
	loss='binary_crossentropy',
	metrics=['acc'])

history = model.fit(train_features, train_labels,
	epochs=30,
	batch_size=20,
	validation_data=(validation_features, validation_labels))