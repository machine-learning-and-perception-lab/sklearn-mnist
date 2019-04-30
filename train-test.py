
from mlp import *

LEARNING_RATE = 0.001
DEPTH = 1

# loading training and text examples
X_train, y_train, X_test, y_test = load_mnist()

# network creation
mlp = create_mlp(LEARNING_RATE, DEPTH)

# training step
#mlp.fit(X_train, y_train)
train_mlp(mlp, X_train, y_train)

# evaluation
test_mlp(mlp, X_test, y_test)

show_plots(mlp)

# fig, axes = plt.subplots(4, 4)
# # use global min / max to ensure all weights are shown on the same scale
# vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
# for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
#     ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
#                vmax=.5 * vmax)
#     ax.set_xticks(())
#     ax.set_yticks(())

# plt.show()
