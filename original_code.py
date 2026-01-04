import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df_salaries = pd.read_csv('salaries.csv')

# Convert categorical data
df_salaries['gender'] = df_salaries['gender'].map({'m': 1, 'f': 0})

# Features and target
X = df_salaries.drop(columns='income')
y = df_salaries['income']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
