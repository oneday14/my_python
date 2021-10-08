from sklearn.datasets import load_iris
iris_dataset = load_iris() 
X = iris_dataset.data
Y = iris_dataset.target

import pandas as pd
data = pd.DataFrame(X, columns = iris_dataset.feature_names)
data['Y'] = Y

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111, projection = '3d')

for color, z, x in zip(['r', 'g', 'b'], [3, 2, 1], [0, 1, 2]):
    xs = np.arange(0, len(data.loc[data.Y == x, 'sepal length (cm)']))
    ys = sorted(data.loc[data.Y == x, 'sepal length (cm)'])

    cs = [color] * len(xs)
    ax.bar(xs, ys, zs = z, zdir = 'y', color = cs, alpha = 0.8)

ax.set_xlabel('sequence')
ax.set_ylabel('Y')
ax.set_zlabel('sepal length (cm)')

plt.title('iris sepal length (cm) distribution')

plt.show()
