from keras import models, layers

from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

# Import VGG16 base, set up dir structure
conv_base = VGG16(weights='imagenet',
				  include_top=False,
				  input_shape=(150, 150, 3))

base_dir = os.join(os.getcwd(), 'data')

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

batch_size = 20

# Initialize Data Generators
train_datagen = ImageDataGenerator(
	rescale=1./255,
	rotation_range=40,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest')

test_datagen = ImageDataGenerator(rescal=1./255)

train_generator = train_datagen.flow_from_directory(
	train_dir,
	target_size=(150, 150),
	batch_size=batch_size,
	class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
	validation_dir,
	target_size=(150, 150),
	batch_size=batch_size,
	class_mode='binary')

# Compile and train model with frozen VGG-16 base
conv_base.trainable = False
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

history = model.fit_generator(
	train_generator,
	steps_per_epoch=100,
	epochs=30,
	validation_data=validation_generator,
	validation_steps=50,
	verbose=2)