import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
st.set_option('deprecation.showPyplotGlobalUse', False)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='white')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    st.pyplot()

def plot_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    st.pyplot()

st.title('Classifier Hyperparameter Tuning and Evaluation')

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Select target variable
    st.subheader("Select Target Variable")
    target_variable = st.selectbox("Select target variable", options=df.columns)

    # Select classifier
    st.subheader("Select Classifier")
    classifier = st.selectbox("Select classifier", options=["Random Forest", "Logistic Regression"])

    # Train-test split
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    if classifier == "Random Forest":
        n_estimators = st.slider("Number of estimators", min_value=10, max_value=200, step=10, value=100)
        max_depth = st.slider("Maximum depth", min_value=1, max_value=20, value=10)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    else:
        C = st.slider("Regularization parameter (C)", min_value=0.01, max_value=10.0, step=0.01, value=1.0)
        model = LogisticRegression(C=C, random_state=42, max_iter=1000)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate model
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=np.unique(y))
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)
    plot_roc_curve(y_test, y_score)
