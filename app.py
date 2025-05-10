import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Split features and target
X = df.drop(['target'], axis=1).values
y = df['target'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Grid search for best Logistic Regression model
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
lr = LogisticRegression(solver='liblinear', random_state=42)
grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# Streamlit UI
st.title("🔬 Breast Cancer Prediction App")
st.write("Enter the patient’s measurements below to predict whether the tumor is **Benign** or **Malignant**.")

# Sidebar inputs
st.sidebar.header("Patient Features Input")
input_data = []
for feature in data.feature_names:
    value = st.sidebar.slider(
        label=feature,
        min_value=float(df[feature].min()),
        max_value=float(df[feature].max()),
        value=float(df[feature].mean())
    )
    input_data.append(value)

# Predict when user clicks button
if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = best_model.predict(input_scaled)
    prediction_proba = best_model.predict_proba(input_scaled)

    st.subheader(" Prediction Result")
    if prediction[0] == 0:
        st.error(f"Result: Malignant (Cancerous)")
    else:
        st.success(f"Result: Benign (Non-cancerous)")

    st.subheader(" Prediction Probability")
    st.write(f"Malignant: {prediction_proba[0][0]:.4f}")
    st.write(f"Benign: {prediction_proba[0][1]:.4f}")