from mlp import *

LEARNING_RATE = 0.01
DEPTH = 1

# loading training and text examples
X_train, y_train, X_test, y_test = load_mnist()

# network creation
mlp = create_mlp(LEARNING_RATE, DEPTH)

# training step
train_accuracy = train_mlp(mlp, X_train, y_train)

# evaluation
test_accuracy = test_mlp(mlp, X_test, y_test)

# show some image examples with predicted labels
show_samples(mlp, X_test, y_test)
