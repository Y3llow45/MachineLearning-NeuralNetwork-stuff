from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

data = load_iris()
X = data.data
Y = data.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

clf = SVC(kernel='linear', C=3)
clf.fit(x_train, y_train)

clf2 = KNeighborsClassifier(n_neighbors=3)
clf2.fit(x_train, y_train)

print("SVM Accuracy:", clf.score(x_test, y_test))
print("KNN Accuracy:", clf2.score(x_test, y_test))
