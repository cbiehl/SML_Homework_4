import re

import numpy as np
from cvxopt import matrix, solvers
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 8.0)


def read_iris_data(filepath):
    with open(filepath, 'r', encoding='utf8') as f:
        x = []
        y = []
        for line in f:
            splitline = re.split('[ ]+', line.strip())
            x.append(np.asarray([float(f) for f in splitline[:-1]]))
            y.append(1.0 if splitline[2] == '0' else -1.0)

        return np.asarray(x), np.asarray(y)


def rbf_kernel(x, z, sigma=.4):
    """gaussian kernel"""
    return np.exp(-(np.linalg.norm(x - z, ord=2) ** 2) / (2 * sigma ** 2))


def plot_data(x, y):
    # TODO: extend to show margin and misclassifications
    plt.figure()
    c = []
    for label in y:
        if label == 1.0:
            c.append('blue')
        else:
            c.append('orange')

    plt.scatter(x[:, 0], x[:, 1], c=c)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Iris dataset")
    # plt.legend()


class SVM:
    def __init__(self, kernel, C, min_sv_alpha=0.00001):
        self.kernel = kernel
        self.C = C
        # because of numpy alphas never get 0
        self.min_sv_alpha = min_sv_alpha

    def fit(self, X, y):
        if type(X) == 'list':
            X = np.asarray(X)

        if type(y) == 'list':
            y = np.asarray(y)

        # evaluate kernel function for all pairs and buffer results
        # (similar to kernel PCA as described by Bishop)
        # TODO: might this one get prohibitively large? Need scipy sparse matrix?
        self.K = np.ndarray(shape=(X.shape[0], X.shape[0]))
        for i, xi in enumerate(X):
            for j, xj in enumerate(X):
                self.K[i, j] = self.kernel(xi, xj)

        # solve lagrangian dual using cvxopt quadratic optimization
        # note that the "matrix" objects are cvxopt objects, not numpy
        # TODO: bias!

        # general SVM
        # Q = matrix(np.outer(y, y) * self.K)
        # p = matrix(np.ones(self.K.shape[0]) * -1)
        # A = matrix(y, (1, self.K.shape[0]))
        # b = matrix(0.0)
        #
        # tmp1 = np.diag(np.ones(self.K.shape[0]) * -1)
        # tmp2 = np.identity(self.K.shape[0])
        # G_x = matrix(np.vstack((tmp1, tmp2)))
        # tmp1 = np.zeros(self.K.shape[0])
        # tmp2 = np.ones(self.K.shape[0]) * self.C
        # h_x = matrix(np.hstack((tmp1, tmp2)))
        #
        # G_slack = matrix(np.diag(np.ones(self.K.shape[0])))
        # h_slack = matrix(np.ones(self.K.shape[0]) * self.C)
        # G = matrix(np.vstack((G_x, G_slack)))
        # h = matrix(np.vstack((h_x, h_slack)))

        Q = matrix(np.outer(y, y) * self.K)
        p = matrix(np.ones(self.K.shape[0]) * -1)
        G_x = matrix(np.diag(np.ones(self.K.shape[0]) * -1))
        h_x = matrix(np.zeros(self.K.shape[0]))
        G_slack = matrix(np.diag(np.ones(self.K.shape[0])))
        h_slack = matrix(np.ones(self.K.shape[0]) * self.C)
        G = matrix(np.vstack((G_x, G_slack)))
        h = matrix(np.vstack((h_x, h_slack)))

        A = matrix(y, (1, self.K.shape[0]))
        b = matrix(0.0)

        solution = solvers.qp(Q, p, G, h, A, b)

        # the solver outputs the alphas (lagrange multipliers)
        self.alphas = np.asarray(solution['x'])[:, 0]
        print('Lagrange multipliers:')
        print(self.alphas)

        # save the support vectors (i.e. the ones with non-zero alphas) and their y values
        sv_index = np.where(self.min_sv_alpha < self.alphas)
        print('\nNumber of support vectors:', sv_index[0].shape[0], end='\n\n')
        self.support_alphas = self.alphas[sv_index]
        self.support_vectors = X[sv_index]
        self.targets = y[sv_index]

        # estimate the bias (Bishop page 334)

        self.b = 0
        for i, yi in enumerate(self.targets):
            m = yi
            for j, xj in enumerate(self.support_vectors):
                m -= self.alphas[j] * self.targets[j] * self.kernel(self.support_vectors[i], xj)
            self.b += m
        self.b /= self.targets.shape[0]

        print('bias:', self.b)

    def predict(self, X):
        scores = np.ndarray(shape=(X.shape[0]))
        for i, xi in enumerate(X):
            for j in range(self.support_vectors.shape[0]):
                scores[i] += self.support_alphas[j] * self.targets[j] * self.kernel(xi, self.support_vectors[j])

            scores[i] += self.b

        return scores


# run experiment on the reduced iris dataset
X, y = read_iris_data('iris-pca.txt')
plot_data(X, y)
svm = SVM(rbf_kernel, C=5)
svm.fit(X, y)
y_hat = svm.predict(X)
pred = np.ndarray(shape=y_hat.shape)

for i, yi in enumerate(y_hat):
    if yi > 0:
        pred[i] = 1.0
    else:
        pred[i] = -1.0

tp = (y == pred).sum()
fn = X.shape[0] - tp
print('Number of misclassified data points:', fn)
print('Accuracy:', tp / X.shape[0])

# plot the result!
plt.figure()
c = []
for i, label in enumerate(y):
    if label == 1.0:
        if pred[i] == label:
            c.append('blue')
        else:
            c.append('fuchsia')
    else:
        if pred[i] == label:
            c.append('orange')
        else:
            c.append('red')

plt.scatter(X[:, 0], X[:, 1], c=c)
# plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1], c="red", marker="x")
# plt.scatter(X[:, 0], X[:, 1], "x", c=c)
# plt.scatter(X[:, 0], X[:, 1], ".", c=c

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Iris dataset with decision boundary")
# plt.show()

steps = 200
x_range = np.linspace(-1.5, 3, steps)
y_range = np.linspace(-3, 3, steps)
x_values, y_values = np.meshgrid(x_range, y_range)
Z = np.zeros((steps, steps))

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
Z = svm.predict(np.c_[x_values.ravel(), y_values.ravel()])

# Put the result into a color plot
Z = Z.reshape(x_values.shape)
plt.contour(x_values, y_values, Z, cmap=plt.cm.Paired)
plt.show()
