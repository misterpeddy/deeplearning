from keras import layers, models
import numpy as np

import os

# Import data 
fname = os.path.join(os.getcwd(), 'data', 'jena_climate_2009_2016.csv')
f = open(fname)
lines = f.read()
f.close()

lines = lines.split('\n')
headers = lines[0].split(',')
lines = lines[1 : ]

float_data = np.zeros((len(lines), len(headers) - 1))
for i, line in enumerate(lines):
	float_data[i, :] = [float(x) for x in line.split(',')[1:]]

mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

# Build generator function
def generator(data, shuffle, step, lookback, min_index, max_index, delay, batch_size=128):
	if max_index is None:
		max_index = len(data) - delay - 1
	i = min_index + lookback
	while 1:
		if shuffle:
			rows = np.random.randint(
				min_index + lookback, max_index, size=batch_size)
		else:
			if (i + batch_size >= max_index):
				i = min_index + lookback
			rows = np.arange(i, min(i + batch_size, max_index))
			i += len(rows)
			
		samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
		targets = np.zeros((len(rows),))
		
		for j, row in enumerate(rows):
			indices = range(rows[j] - lookback, rows[j], step)
			samples[j] = data[indices]
			targets[j] = data[rows[j] + delay][1]

		yield (samples, targets)

# Create training, validation and test generators

lookback = 1440
delay = 144
batch_size = 128
val_steps = 300000 - 200000 - lookback
gru_width = 32

train_gen = generator(float_data, 
					shuffle=True,
					step=6,
					lookback=lookback,
					min_index=0,
					max_index=200000,
					delay=delay,
					batch_size=batch_size)

validation_gen = generator(float_data, 
					shuffle=True,
					step=6,
					lookback=lookback,
					min_index=200001,
					max_index=300000,
					delay=delay,
					batch_size=batch_size)

test_gen = generator(float_data, 
					shuffle=True,
					step=6,
					lookback=lookback,
					min_index=300000,
					max_index=None,
					delay=delay,
					batch_size=batch_size)

model = models.Sequential()
model.add(layers.Bidirectional(
	layers.GRU(
		gru_width,
		dropout=0.1,
		recurrent_dropout=0.5,
		activation='relu'),
	input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer='rmsprop', loss='mae', metrics=['mae'])

history = model.fit_generator(train_gen,
				steps_per_epoch=500,
				epochs=20,
				validation_data=validation_gen,
				validation_steps = val_steps)

loss = history['loss']
val_loss = history['val_loss']

epochs = np.arange(1, len(loss))

plt.plot(epochs, loss, 'bo', title="Training Loss")
plt.plot(epochs, val_loss, 'b', title="Validation Loss")
