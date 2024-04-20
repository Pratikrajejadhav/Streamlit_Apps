import streamlit as st
import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
st.set_option('deprecation.showPyplotGlobalUse', False)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns 
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,auc
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

st.title('**Logistic Regression Hyperparameter Tuning and Evaluation**')

#upload a file
file_upload = st.sidebar.file_uploader("**Upload Only Preprocessed CSV File**",type=['csv'])

if file_upload is not None:
    data = pd.read_csv(file_upload)

    # select target column
    target_column = st.selectbox("**Select The Target Column First**",options=data.columns)

    # input output column
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Logistic Regression hyperparameters
    st.sidebar.header("**Hyperparameters of Logistic Regression**")
    penalty = st.sidebar.selectbox("**Penalty**", options=["l1", "l2"],index=1)
    solver = st.sidebar.selectbox("**Solver**", options=["newton-cg", "lbfgs", "liblinear", "sag", "saga"],index=2)
    C = st.sidebar.slider("**Regularization Strength (C)**",min_value=0.1,max_value=10.0,step=0.1,value=0.5)
    max_iter = st.sidebar.slider("**Maximum Number of Iterations**",min_value= 100 ,max_value= 1000 ,step=100 ,value=300)

    LR = LogisticRegression(penalty=penalty, solver=solver,C=C, max_iter=max_iter)


    # Model training
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
     
    #calulate accuracy,precision,recall,f1_score 
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    

    st.write("Accuracy Score :",accuracy)
    st.write("Precision Score :", precision)
    st.write("Recall Score :", recall)
    st.write("F1 Score :", f1)

    # plot confusion matrix
    cm = confusion_matrix(y_test,y_pred)
    display = ConfusionMatrixDisplay(cm,display_labels=[False,True])
    display.plot()
    plt.grid(False)
    plt.show()
    st.pyplot()
 
    tn,fp,fn,tp = confusion_matrix(y_test,y_pred).ravel()
    st.write("True Positive :",tp)
    st.write("True Negative :",tn)
    st.write("False Positive :",fp)
    st.write("False Negative :",fn)

    #Plot Roc curve
    y_pred_prob=LR.predict_proba(X_test)[:,1]   # Class-1 probabilities
    fpr,tpr,threshold=roc_curve(y_test,y_pred_prob) 
    plt.plot([0,1],[0,1],color="red",lw=2,label="Average-model")
    plt.plot(fpr,tpr,color="orange",lw=2,label="Random Forest Model with Hyperparameter")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic :ROC-AUC")
    plt.legend()
    plt.show()
    st.pyplot()

  
    #Area under the curve
    st.write("Computed Area Under the Curve (AUC)",(auc(fpr, tpr)))
    

