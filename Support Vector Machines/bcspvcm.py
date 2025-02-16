import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
Y = data.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

models = {
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "RandomForest": RandomForestClassifier()
}

param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(x_train, y_train)

for name, model in models.items():
    if name == "SVM":
        model = grid_search.best_estimator_ 
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(f"{name} Accuracy: {acc:.4f}")

print("Best SVM parameters:", grid_search.best_params_)
