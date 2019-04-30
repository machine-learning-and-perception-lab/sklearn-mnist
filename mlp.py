from warnings import filterwarnings
filterwarnings('ignore')

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

def load_mnist():
	# Load data from https://www.openml.org/d/554
	print('Loading MNIST dataset...')
	
	X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
	X = X / 255.

	# rescale the data, use the traditional train/test split
	X_train, X_test = X[:60000], X[60000:]
	y_train, y_test = y[:60000], y[60000:]

	print('MNIST dataset loaded!')

	return X_train, y_train, X_test, y_test

def create_mlp(learning_rate, depth):
	print('Building MLP...')

	hidden_layer_sizes = tuple([64 for _ in range(depth)])
	epochs = 5#50
	solver = 'adam'
	weight_decay = 1e-4

	mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, 
						max_iter=epochs, 
						alpha=weight_decay,
                    	solver=solver, 
                    	verbose=True, 
                    	random_state=123,
                    	learning_rate_init=learning_rate)

	print('MLP built!')

	return mlp

def train_mlp(mlp, X_train, y_train):
	print('Training MLP...')

	mlp.fit(X_train, y_train)

	print('Training done!')

	print("Training set accuracy: %f" % mlp.score(X_train, y_train))

def test_mlp(mlp, X_test, y_test):
	print("Test set accuracy: %f" % mlp.score(X_test, y_test))

def show_plots(mlp):
	plt.plot(mlp.loss_curve_)
	plt.title('Loss')
	plt.ylabel('Loss value')
	plt.xlabel('Epochs')
	plt.show()
