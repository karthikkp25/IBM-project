import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
df = pd.read_csv('data.csv')

# One-hot encode 'Place'
df_encoded = pd.get_dummies(df, columns=['Place'], drop_first=True)

# Separate features and target
X = df_encoded.drop("Price (in Cr)", axis=1)
y = df_encoded["Price (in Cr)"]

# Save column names used in training (for Streamlit app)
joblib.dump(X.columns.tolist(), "model_columns.pkl")

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])

# Fit pipeline and transform data
X_train_prepared = preprocessing_pipeline.fit_transform(X_train)
X_val_prepared = preprocessing_pipeline.transform(X_val)

# Train best model (Random Forest)
best_model = RandomForestRegressor()
best_model.fit(X_train_prepared, y_train)

# Evaluate model
y_pred = best_model.predict(X_val_prepared)
print("R2 Score:", r2_score(y_val, y_pred))
print("MSE:", mean_squared_error(y_val, y_pred))

# Save model and pipeline
joblib.dump(best_model, "best_model.pkl")
joblib.dump(preprocessing_pipeline, "preprocessing_pipeline.pkl")
