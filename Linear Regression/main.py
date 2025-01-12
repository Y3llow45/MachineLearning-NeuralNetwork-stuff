import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

time_studied = np.array([4, 7, 11, 19, 21, 24, 28, 33, 37, 45, 51, 57, 65]).reshape(-1, 1)
scores = np.array([25, 47, 34, 38, 54, 42, 56, 47, 76, 84, 87, 60, 92]).reshape(-1, 1)

model = LinearRegression()
model.fit(time_studied, scores)

print(model.predict(np.array([39]).reshape(-1, 1)))

plt.scatter(time_studied, scores)
plt.plot(np.linspace(0, 70, 100).reshape(-1, 1), model.predict(np.linspace(0, 70, 100).reshape(-1, 1)), 'r')
plt.ylim(0, 100)
plt.show()
