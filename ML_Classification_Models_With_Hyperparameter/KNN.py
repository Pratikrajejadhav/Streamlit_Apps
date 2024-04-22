import streamlit as st 
import pandas as pd
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score,recall_score,accuracy_score,roc_auc_score,auc,f1_score,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("KNN with Hyperparameter Tunning and Evaluation")

#upload a file
file_upload = st.sidebar.file_uploader("**Upload Only Preprocessed CSV File**",type=['csv'])

if file_upload is not None:
    data = pd.read_csv(file_upload)

    #select target column 
    target_column = st.selectbox("**Choose Your Target Coulnm**",options=data.columns)

    # input output column
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # train test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression hyperparameters
    st.sidebar.header("**Hyperparameters of KNN **")
    leaf_size = st.sidebar.slider("**Leaf Size**",min_value=10,max_value=100,step=10,value=30)
    n_neighbours = st.sidebar.slider("**No of Neighburs**",min_value=1,max_value=10,step=1,value=3)
    p = st.sidebar.slider("**P**",min_value=1,max_value=5,step=1,value=2)
    matrix = st.sidebar.selectbox("**Distance Matrix**", options=["minkowski", "manhattan","euclidean","chebyshev","cosine"],index=0)
    
    #load the model
    KNN = KNeighborsClassifier(p=p,leaf_size=leaf_size,n_neighbors=n_neighbours,metric=matrix,n_jobs=-1)

    
    # Model training
    KNN.fit(X_train, y_train)

    #predict the model
    y_pred = KNN.predict(X_test)
     
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

    # plot auc-roc curve
    y_pred_prob=KNN.predict_proba(X_test)[:,1]   # Class-1 probabilities
    fpr,tpr,threshold=roc_curve(y_test,y_pred_prob) 
    plt.plot([0,1],[0,1],color="red",lw=3,label="Average-model")
    plt.plot(fpr,tpr,color="orange",lw=3,label="Random Forest Model with Hyperparameter")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic :ROC-AUC")
    plt.legend()
    plt.show()
    st.pyplot()

    #Area under the curve
    st.write("Computed Area Under the Curve (AUC)",(auc(fpr, tpr)))
 
    
    
    