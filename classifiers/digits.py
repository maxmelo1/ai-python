from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

digits = datasets.load_digits()

fig, axis = plt.subplots(2, 10)

for i in range(0, 2):
    for j in range(0,10):
        current = 20 - (i*10+j)
        axis[i, j].imshow(digits.images[-current], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

print(digits.target[0])

mlp = MLPClassifier(hidden_layer_sizes=(50, ), activation='logistic', max_iter=100, alpha=0.001, solver='sgd', tol=1e-9, learning_rate_init=0.1, verbose=True)

mlp.fit(digits.data[:-20], digits.target[:-20])

print(mlp.predict(digits.data[-20:]))
print(digits.target[-20:])
