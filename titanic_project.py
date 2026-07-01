import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Fill missing values
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())

train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

# Convert text to numbers
train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})

train["Embarked"] = train["Embarked"].map({"S": 0, "C": 1, "Q": 2})
test["Embarked"] = test["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Select features
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

X = train[features]
y = train["Survived"]

X_test = test[features]

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Make predictions
predictions = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})

submission.to_csv("submission.csv", index=False)

print("✅ Submission file created successfully!")