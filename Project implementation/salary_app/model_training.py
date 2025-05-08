


# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import pickle

# 1. Load Dataset
df = pd.read_csv('salary.csv')

# 2. Encode Categorical Columns
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# 3. Features & Target
X = df.drop(columns=['Estimated_Annual_Salary'])
y = df['Estimated_Annual_Salary']

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model + Grid Search
xgb = XGBRegressor(random_state=42)
params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.3]
}

grid = GridSearchCV(xgb, params, cv=3, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# 6. Evaluation
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("✅ Model Performance")
print(f"MAE: ₹{mae:.2f}")
print(f"RMSE: ₹{rmse:.2f}")
print(f"R²: {r2:.2f}")
print(f"Best Parameters: {grid.best_params_}")

# 7. Save Model
with open("salary_model.pkl", "wb") as f:
    pickle.dump({
        "model": best_model,
        "label_encoders": label_encoders,
        "feature_columns": X.columns.tolist()
    }, f)



# %%


# %%



