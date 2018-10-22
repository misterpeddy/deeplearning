from keras import models, layers
from keras.datasets import boston_housing

import matplotlib.pyplot as plt
import numpy as np

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

NUM_VALIDATION_FOLDS = 4
NUM_EPOCHS = 100
VERBOSE = 0

# Do feature-wise normalization
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data -= mean
train_data /= std
test_data -= mean
test_data /= std

# Function to build model
def build_model():
  model = models.Sequential()
  model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(1))
  model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
  return model

# Do K-Fold Validation
fold_size = len(train_data) // NUM_VALIDATION_FOLDS
all_scores = []
all_mae_histories = []

for i in range(NUM_VALIDATION_FOLDS):
  print("processing fold #", i)
  
  val_data = train_data[i * fold_size: (i+1) * fold_size]
  val_targets = train_targets[i * fold_size: (i+1) * fold_size]
  
  partial_training_data = np.concatenate(
      [train_data[:i * fold_size], 
      train_data[(i+1) * fold_size:]], axis=0)
  partial_training_targets = np.concatenate(
      [train_targets[:i * fold_size], 
      train_targets[(i+1) * fold_size:]], axis=0)
  
  model = build_model()
  history = model.fit(partial_training_data, 
                      partial_training_targets, 
                      validation_data=(val_data, val_targets),
                      epochs=NUM_EPOCHS, 
                      batch_size=1, 
                      verbose=VERBOSE)
  
  mae_history = history.history['val_mean_absolute_error']
  all_mae_histories.append(mae_history)

# Calculate average MAE for each epoch
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(NUM_EPOCHS)]

# Cut out first 10 epochs and implement explonential moving average
def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1] 
      smoothed_points.append(previous * factor + point * (1-factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

smoothed_mae_history = smooth_curve(average_mae_history[10:])

# Plot the MAE over epochs after 10
plt.plot(range(10, len(smoothed_mae_history) + 10), smoothed_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
