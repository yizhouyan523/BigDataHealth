import matplotlib.pyplot as plt
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	fig, axs = plt.subplots(1,2,figsize=(20,10))
	axs[0].set_title('Loss Curves')
	axs[0].plot(train_losses, 'C0', label ='Training Loss')
	axs[0].plot(valid_losses, 'C1', label ='Validation Loss')
	print('plt',train_losses)
	axs[0].legend(loc ="upper right")
	axs[0].set_xlabel("Epoch")
	axs[0].set_ylabel("Loss")
	axs[1].set_title('Accuracy Curves')
	axs[1].plot(train_accuracies, 'C0', label ='Training Accuracy')
	axs[1].plot(valid_accuracies, 'C1', label ='Validation Accuracy')
	axs[1].legend(loc ="upper left")
	axs[1].set_xlabel("Epoch")
	axs[1].set_ylabel("Accuracy")
	fig.savefig('Learning_Curve.png')


def plot_confusion_matrix(results, class_names):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	y_true, y_pred = zip(* results)
	Matrix = confusion_matrix(y_true,y_pred)
	np.set_printoptions(precision = 2)
	plt.figure(figsize=(10,10))
	Matrix = Matrix.astype('float32')/Matrix.sum(axis=1)[:,np.newaxis]
	plt.imshow(Matrix, interpolation='nearest', cmap=plt.cm.Blues)
	plt.colorbar()
	for x in range(Matrix.shape[0]):
		for y in range(Matrix.shape[1]):
			plt.text(y, x, format(Matrix[x, y], '.2f'), fontsize=12,horizontalalignment = "center", color = "white" if Matrix[x, y] > Matrix.max()/2 else "black")
	plt.xticks(np.arange(0,len(class_names),1),class_names, rotation=45)
	plt.yticks(np.arange(0,len(class_names),1),class_names)
	plt.title('Normalized Confusion Matrix')
	plt.ylabel('True')
	plt.xlabel('Predicted')
	plt.savefig('confusion_matrix.png')











