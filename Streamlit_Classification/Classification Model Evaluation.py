import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
st.set_option('deprecation.showPyplotGlobalUse', False)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    st.pyplot()

# Function to plot ROC curve
def plot_roc_curve(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    st.pyplot()

# Streamlit UI
st.title('Classification Model Evaluation')
st.subheader("Upload A Preprocessed And Standardize Data file")
# File uploader
uploaded_file = st.file_uploader("Upload Data file", type=["csv"])

if uploaded_file is not None:
    # Read data
    df = pd.read_csv(uploaded_file)

    # Show data
    st.write(df)

    # Select target column
    target_column = st.sidebar.selectbox("Select target column", df.columns)

    # Select algorithm
    algorithm = st.sidebar.selectbox("Select classification algorithm", ["KNN", "Naive Bayes", "Decision Tree","Logistic Regression"])

    # Prepare data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    if algorithm == "KNN":
        model = KNeighborsClassifier()
    elif algorithm == "Naive Bayes":
        model = GaussianNB()
    elif algorithm == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_prob)
    
    # Accuracy 
    accuracy = accuracy_score(y_test, y_pred)

    st.markdown(f"### **Accuracy Score : {accuracy}**")