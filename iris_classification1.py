"""CodeAlpha Task 1: Iris Flower Classification"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. Load data (built into scikit-learn; or use pd.read_csv("Iris.csv"))
iris = load_iris(as_frame=True)
df = iris.frame
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
print(df.head(), "\n")
print(df.describe(), "\n")
print("Missing values:\n", df.isnull().sum(), "\n")

# 2. EDA
sns.pairplot(df.drop(columns="target"), hue="species")
plt.savefig("iris_pairplot.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. Split
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Compare models
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": make_pipeline(StandardScaler(), SVC()),
}
results = {}
for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train, cv=5).mean()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results[name] = (cv, acc)
    print(f"{name:20s} CV acc: {cv:.3f} | Test acc: {acc:.3f}")

# 5. Evaluate best model
best_name = max(results, key=lambda k: results[k][1])
best = models[best_name]
pred = best.predict(X_test)
print(f"\nBest model: {best_name}")
print(classification_report(y_test, pred, target_names=iris.target_names))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=iris.target_names).plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}")
plt.savefig("iris_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# 6. Predict a new flower
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=iris.feature_names)
print("Prediction for sample:", iris.target_names[best.predict(sample)[0]])