import matplotlib.pyplot as plt

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
