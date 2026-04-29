# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("SuperMarket Analysis.csv")

# Display dataset information
print("Dataset Preview:")
print(df.head())

print("\nDataset Information:")
print(df.info())

# -------------------------------------
# Feature Engineering
# -------------------------------------

# Convert Date column and extract features
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.weekday

# Drop Date and Time columns (not usable for ML models)
if 'Date' in df.columns:
    df.drop(columns=['Date'], inplace=True)

if 'Time' in df.columns:
    df.drop(columns=['Time'], inplace=True)

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Create interaction feature
if 'Unit price' in df.columns and 'Quantity' in df.columns:
    df['Price_Interaction'] = df['Unit price'] * df['Quantity']

# Drop unnecessary column
if 'Invoice ID' in df.columns:
    df.drop(columns=['Invoice ID'], inplace=True)

# -------------------------------------
# Define Features and Target
# -------------------------------------

target = 'Quantity'

X = df.drop(columns=[target])
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------
# Models
# -------------------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
}

# -------------------------------------
# Training and Evaluation
# -------------------------------------

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

    results[name] = {
        "MSE": mse,
        "R2 Score": r2,
        "Cross-Validation R2": np.mean(cv_scores)
    }

    print(f"{name} Performance:")
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")
    print(f"Cross-Validation R2 (Mean): {np.mean(cv_scores)}")

# -------------------------------------
# Model Comparison
# -------------------------------------

results_df = pd.DataFrame(results).T

print("\nModel Comparison:")
print(results_df)

results_df[['R2 Score', 'Cross-Validation R2']].plot(kind='bar')
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# -------------------------------------
# Price Optimization
# -------------------------------------

if 'Unit price' in X.columns:
    sample = X_test.iloc[0].copy()

    price_range = np.linspace(df['Unit price'].min(), df['Unit price'].max(), 50)
    predicted_demand = []

    best_model = models["Random Forest Regressor"]

    for price in price_range:
        temp = sample.copy()
        temp['Unit price'] = price
        prediction = best_model.predict([temp])[0]
        predicted_demand.append(prediction)

    revenue = price_range * predicted_demand

    # Demand vs Price
    plt.figure()
    plt.plot(price_range, predicted_demand)
    plt.title("Demand vs Price")
    plt.xlabel("Price")
    plt.ylabel("Predicted Demand")
    plt.tight_layout()
    plt.show()

    # Revenue vs Price
    plt.figure()
    plt.plot(price_range, revenue)
    plt.title("Revenue vs Price")
    plt.xlabel("Price")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.show()

    # Optimal Price
    optimal_price = price_range[np.argmax(revenue)]

    print(f"\nOptimal Price for Maximum Revenue: {optimal_price}")