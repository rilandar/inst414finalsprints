###Make sure to install the required libraries: matplotlib, pandas,&sklearn###


################Random Forest Regression for Retail Sales Prediction#######################
# Random Forest Regressor for Revenue Prediction (combined graph)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load and sort data
file_path = r'C:\Users\Kanis\OneDrive\Documents\EnergyPredictingML\data\processed\mllearningdata.csv'
df = pd.read_csv(file_path).sort_values(by='Year').reset_index(drop=True)

# ----------------- Random Forest: Total Retail Sales -----------------
years = [2021, 2022, 2023]
predicted_sales = []
actual_sales = []

for year in years:
    train_data = df[df["Year"] < year]
    X_train = train_data[["Avg Price(cents/kWh)"]]
    y_train = train_data["Total retail sales(MWh)"]
    
    test_data = df[df["Year"] == year][["Avg Price(cents/kWh)"]]
    actual_sales.append(df[df["Year"] == year]["Total retail sales(MWh)"].values[0])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(test_data)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    predicted_sales.append(model.predict(X_test_scaled)[0])

plt.figure(figsize=(10, 6))
x = np.arange(len(years))
width = 0.4
plt.bar(x - width/2, actual_sales, width, label="Actual Sales", color='blue')
plt.bar(x + width/2, predicted_sales, width, label="Predicted Sales", color='red')
plt.xlabel("Year")
plt.ylabel("Total Retail Sales(MWh)")
plt.title("Actual vs Predicted Total Retail Sales")
plt.xticks(ticks=x, labels=years)
plt.legend()
for i in range(len(years)):
    plt.text(i - width/2, actual_sales[i] + 500000, f"{actual_sales[i]:,.0f}", ha='center')
    plt.text(i + width/2, predicted_sales[i] + 500000, f"{predicted_sales[i]:,.0f}", ha='center')
plt.tight_layout()
plt.show()

# ----------------- Random Forest: Revenue -----------------
predictions = []
actuals = []
for year in years:
    train_data = df[df["Year"] <= year - 1]
    X_train = train_data[["Avg Price(cents/kWh)", "Total retail sales(MWh)"]]
    y_train = train_data["Revenue(thousand dollars)"]
    
    test_data = df[df["Year"] == year][["Avg Price(cents/kWh)", "Total retail sales(MWh)"]]
    actual_revenue = df[df["Year"] == year]["Revenue(thousand dollars)"].values[0]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(test_data)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    predictions.append(model.predict(X_test_scaled)[0])
    actuals.append(actual_revenue)

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, actuals, width, label='Actual Revenue', color='blue')
plt.bar(x + width/2, predictions, width, label='Predicted Revenue', color='orange')
plt.xlabel("Year")
plt.ylabel("Revenue (thousand dollars)")
plt.title("Actual vs Predicted Revenue (2021–2023)")
plt.xticks(x, years)
plt.legend()
for i in range(len(years)):
    plt.text(x[i] - width/2, actuals[i] + 50000, f"{actuals[i]:,.0f}", ha='center')
    plt.text(x[i] + width/2, predictions[i] + 50000, f"{predictions[i]:,.0f}", ha='center')
plt.tight_layout()
plt.show()

# ----------------- Gradient Boosting: Revenue and Retail -----------------
targets = ["Revenue(thousand dollars)", "Total retail sales(MWh)"]
results = {t: {"actual": [], "predicted": []} for t in targets}

for i in range(3, 6):
    train_df = df.iloc[:i]
    test_df = df.iloc[[i]]
    for target in targets:
        X_train = train_df.drop(columns=targets)
        y_train = train_df[target]
        X_test = test_df.drop(columns=targets)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = GradientBoostingRegressor(random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        results[target]["actual"].append(test_df[target].values[0])
        results[target]["predicted"].append(y_pred[0])

for target in targets:
    actual = results[target]["actual"]
    predicted = results[target]["predicted"]
    plt.figure(figsize=(8, 5))
    plt.plot(years, actual, marker='o', label="Actual", linewidth=2)
    plt.plot(years, predicted, marker='x', label="Predicted", linestyle='--', linewidth=2)
    plt.title(f"Actual vs Predicted {target}")
    plt.xlabel("Year")
    plt.ylabel(target)
    plt.xticks(years)
    for x, y in zip(years, actual):
        plt.text(x, y, f'{y:,.0f}', va='bottom', ha='center', fontsize=9, color='blue')
    for x, y in zip(years, predicted):
        plt.text(x, y, f'{y:,.0f}', va='top', ha='center', fontsize=9, color='red')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
