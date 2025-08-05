import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the CSV
df = pd.read_csv("dogs_vs_cats_sample.csv")

# Step 2: Split features and label
X = df.drop("label", axis=1)
y = df["label"]

# Step 3: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
