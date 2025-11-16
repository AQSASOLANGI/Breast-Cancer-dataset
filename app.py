# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="Breast Cancer PCA & ML", layout="wide")
st.title("Breast Cancer Classification with PCA & Machine Learning")

# =========================
# Load Dataset
# =========================
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

# Convert to DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

st.subheader("Dataset Preview")
st.dataframe(df.head())

# =========================
# Sidebar - Model Selection
# =========================
st.sidebar.header("Model Selection & Settings")
model_choice = st.sidebar.selectbox("Choose a Model", ("Logistic Regression", "Random Forest", "SVM", "K-Nearest Neighbors"))
pca_components = st.sidebar.slider("Number of PCA Components", min_value=2, max_value=X.shape[1], value=10)

# =========================
# Standardize Data
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# Apply PCA
# =========================
pca = PCA(n_components=pca_components)
X_pca = pca.fit_transform(X_scaled)

st.subheader("Explained Variance by PCA")
cum_variance = np.cumsum(pca.explained_variance_ratio_)
fig, ax = plt.subplots()
ax.plot(range(1, len(cum_variance)+1), cum_variance, marker='o')
ax.set_xlabel("Number of Components")
ax.set_ylabel("Cumulative Explained Variance")
ax.set_title("Cumulative Explained Variance")
ax.grid(True)
st.pyplot(fig)

# =========================
# Train-Test Split
# =========================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

# =========================
# Train Selected Model
# =========================
def get_model(choice):
    if choice == "Logistic Regression":
        return LogisticRegression(max_iter=500)
    elif choice == "Random Forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif choice == "SVM":
        return SVC(kernel='linear', probability=True)
    elif choice == "K-Nearest Neighbors":
        return KNeighborsClassifier(n_neighbors=5)

model = get_model(model_choice)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =========================
# Evaluation
# =========================
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.subheader("Model Evaluation")
st.write(f"**Selected Model:** {model_choice}")
st.write(f"**Accuracy:** {acc:.4f}")

st.subheader("Confusion Matrix")
fig_cm, ax_cm = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap=plt.cm.Blues, ax=ax_cm)
st.pyplot(fig_cm)

st.subheader("Classification Report")
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# =========================
# PCA Scatter Plot
# =========================
if pca_components >= 2:
    st.subheader("PCA Scatter Plot (First 2 Components)")
    plt.figure(figsize=(8,6))
    for target_value, target_name in enumerate(target_names):
        subset = X_pca[y == target_value]
        plt.scatter(subset[:,0], subset[:,1], label=target_name, alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Breast Cancer PCA Scatter Plot")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
