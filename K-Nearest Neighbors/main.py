import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

points = {
    'blue': [[2, 4], [1, 3], [2, 3], [3, 2], [2, 1]],
    'orange': [[5, 6], [4, 5], [4, 6], [6, 6], [5, 4]],
    'green': [[7, 8], [8, 7], [7, 7], [8, 8], [9, 7]],
    'red': [[1, 8], [2, 7], [1, 7], [2, 8], [3, 7]]
}

test = [[3, 3], [9, 8], [3, 8], [5, 5]]

class K_Nearest_Neighbors:
    def __init__(self, k=3):
        self.k = k

    def fit(self, points):
        self.points = points

    def euclidean_distance(self, p, q):
        return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))

    def predict(self, new_point):
        distances = []
        for category in self.points:
            for point in self.points[category]:
                distance = self.euclidean_distance(point, new_point)
                distances.append((distance, category))
        distances.sort(key=lambda x: x[0])
        nearest = [category for _, category in distances[:self.k]]
        return Counter(nearest).most_common(1)[0][0]

knn = K_Nearest_Neighbors(k=3)
knn.fit(points)

for point in test:
    print(f"Point {point} is classified as {knn.predict(point)}")

ax = plt.subplot()
colors = {'blue': 'b', 'orange': 'orange', 'green': 'g', 'red': 'r'}

for category in points:
    for point in points[category]:
        ax.scatter(point[0], point[1], c=colors[category], label=category)

for point in test:
    ax.scatter(point[0], point[1], c='black', marker='x')

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

plt.show()
