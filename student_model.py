import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("student_performance.csv")

print(data.head())
print(data.info())
print(data.describe())

# Features and target//
X = data[['weekly_self_study_hours', 'attendance_percentage', 'class_participation']]
y = data['total_score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Error
error = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", error)

# Graph
plt.scatter(y_test, predictions)
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted Scores")
plt.show()