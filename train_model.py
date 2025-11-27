import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

try:
    df = pd.read_csv('dataset/House_Rent_Dataset.csv')
    print("✅ Data Loaded Successfully!")
except FileNotFoundError:
    print("❌ Error: Could not find 'dataset/House_Rent_Dataset.csv'")
    print("Please create a 'dataset' folder and put the Kaggle CSV inside it.")
    exit()

def clean_floor(x):
    try:
        if 'Ground' in x: return 0
        else: return int(x.split(' ')[0])
    except: return 0

df['Floor_Level'] = df['Floor'].apply(clean_floor)
df = df[df['Rent'] < 500000] # Remove outliers

features = ['BHK', 'Size', 'Floor_Level', 'City', 'Furnishing Status', 'Bathroom']
target = 'Rent'
X = df[features]
y = df[target]

numeric_features = ['BHK', 'Size', 'Floor_Level', 'Bathroom']
categorical_features = ['City', 'Furnishing Status']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1))
])

print("Training Model... Please wait.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

score = r2_score(y_test, model_pipeline.predict(X_test))
print(f"Model Trained! Accuracy Score: {score:.4f}")

import os
if not os.path.exists('models'):
    os.makedirs('models')
    
with open('models/rent_model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)
print("Model saved to 'models/rent_model.pkl'")
