import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
salary_data = pd.read_csv('salaries.csv')

# Encode categorical column
salary_data['gender'] = salary_data['gender'].map({'m': 1, 'f': 0})

# Separate features and target
features = salary_data.drop(columns='income')
target = salary_data['income']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
predicted_income = lr_model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, predicted_income)
r2 = r2_score(y_test, predicted_income)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Visualization
plt.scatter(y_test, predicted_income)
plt.xlabel("Actual Income")
plt.ylabel("Predicted Income")
plt.title("Actual vs Predicted Income")
plt.show()
