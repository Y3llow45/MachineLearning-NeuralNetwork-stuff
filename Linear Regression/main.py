import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

time_studied = np.array([20, 51, 33, 65, 22, 45, 11, 7, 21, 34, 25, 4, 57]).reshape(-1, 1)
scores = np.array([55, 85, 46, 92, 49, 83, 46, 77, 54, 79, 53, 67, 56]).reshape(-1, 1)

model = LinearRegression()
model.fit(time_studied, scores)

print(model.predict(np.array([39]).reshape(-1, 1)))

plt.scatter(time_studied, scores)
plt.plot(np.linspace(0, 70, 100).reshape(-1, 1), model.predict(np.linspace(0, 70, 100).reshape(-1, 1)), 'r')
plt.ylim(0, 100)
plt.show()
