import numpy as np
#from cvxopt import matrix, solvers
import re
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
    
    
def rbf_kernel(x, z, sigma):
    return np.exp(-(np.linalg.norm(x - z, ord=2) ** 2) / (2 * sigma**2))

def plot_data(x, y):
    #TODO: extend to show margin and misclassifications
    plt.figure()
    c = []
    for label in y:
        if label == 1.0:
            c.append('blue')
        else:
            c.append('orange')
            
    plt.scatter(x[:,0], x[:,1], c=c)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Iris dataset")
    #plt.legend()
    
x, y = read_iris_data('iris-pca.txt')
plot_data(x, y)