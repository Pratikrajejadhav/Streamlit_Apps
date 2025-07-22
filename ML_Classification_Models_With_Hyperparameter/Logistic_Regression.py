# import streamlit as st
# import pandas as pd 
# import numpy as np
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# import seaborn as sns 
# from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,auc
# from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
# import matplotlib.pyplot as plt

# st.title('**Logistic Regression Hyperparameter Tuning and Evaluation**')

# #upload a file
# file_upload = st.sidebar.file_uploader("**Upload Only Preprocessed CSV File**",type=['csv'])

# if file_upload is not None:
#     data = pd.read_csv(file_upload)

#     # select target column
#     target_column = st.selectbox("**Select The Target Column First**",options=data.columns)

#     # input output column
#     X = data.drop(columns=[target_column])
#     y = data[target_column]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Logistic Regression hyperparameters
#     st.sidebar.header("**Hyperparameters of Logistic Regression**")
#     penalty = st.sidebar.selectbox("**Penalty**", options=["l1", "l2"],index=1)
#     solver = st.sidebar.selectbox("**Solver**", options=["newton-cg", "lbfgs", "liblinear", "sag", "saga"],index=2)
#     C = st.sidebar.slider("**Regularization Strength (C)**",min_value=0.1,max_value=10.0,step=0.1,value=0.5)
#     max_iter = st.sidebar.slider("**Maximum Number of Iterations**",min_value= 100 ,max_value= 1000 ,step=100 ,value=300)

#     LR = LogisticRegression(penalty=penalty, solver=solver,C=C, max_iter=max_iter)


#     # Model training
#     LR.fit(X_train, y_train)
#     y_pred = LR.predict(X_test)
     
#     #calulate accuracy,precision,recall,f1_score 
#     accuracy = accuracy_score(y_test,y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
    

#     st.write("Accuracy Score :",accuracy)
#     st.write("Precision Score :", precision)
#     st.write("Recall Score :", recall)
#     st.write("F1 Score :", f1)

#     # plot confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
#     display = ConfusionMatrixDisplay(cm, display_labels=[False, True])
#     display.plot()
#     plt.grid(False)
#     fig1 = plt.gcf()
#     st.pyplot(fig1)

 
#     tn,fp,fn,tp = confusion_matrix(y_test,y_pred).ravel()
#     st.write("True Positive :",tp)
#     st.write("True Negative :",tn)
#     st.write("False Positive :",fp)
#     st.write("False Negative :",fn)

#     #Plot Roc curve
#     y_pred_prob = LR.predict_proba(X_test)[:, 1]
#     fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)
#     plt.plot([0, 1], [0, 1], color="red", lw=2, label="Average-model")
#     plt.plot(fpr, tpr, color="orange", lw=2, label="Logistic Regression Model")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("Receiver Operating Characteristic (ROC-AUC)")
#     plt.legend()
#     fig2 = plt.gcf()
#     st.pyplot(fig2)

  
#     #Area under the curve
#     st.write("Computed Area Under the Curve (AUC)",(auc(fpr, tpr)))
    

# -----------------------------------------------------------------------------------------------------------------
#----------------------pre------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Logistic Regression Trainer", layout="wide")
st.title('🧠 Logistic Regression Hyperparameter Tuning and Evaluation')

# Upload CSV
file_upload = st.sidebar.file_uploader("📁 Upload CSV File", type=['csv'])

if file_upload is not None:
    data = pd.read_csv(file_upload)

    st.subheader("🔍 Select Columns")

    # Optional ID column drop
    id_column = st.selectbox("📌 Select ID Column to Drop (Optional)", ["None"] + list(data.columns))
    if id_column != "None":
        data = data.drop(columns=[id_column])
        st.info(f"✅ Dropped ID column: `{id_column}`")

    # Select target
    target_column = st.selectbox("🎯 Select Target Column", options=data.columns)

    # Handle missing target values
    if data[target_column].isnull().any():
        st.warning(f"⚠️ Dropped {data[target_column].isnull().sum()} rows with missing target values.")
        data = data.dropna(subset=[target_column])

    # Split data
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Encode binary target
    if y.dtype == 'object':
        unique_vals = y.unique()
        if len(unique_vals) == 2:
            y = y.map({unique_vals[0]: 0, unique_vals[1]: 1})
            st.success(f"✅ Encoded target: {unique_vals[0]} → 0, {unique_vals[1]} → 1")
        else:
            st.error("❌ Target column must be binary for logistic regression.")
            st.stop()

    # ===============================
    # 🔧 Preprocessing
    # ===============================
    st.subheader("⚙️ Data Preprocessing")

    # Fill missing values
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna(X[col].mode()[0])
        else:
            X[col] = X[col].fillna(X[col].median())
    st.success("✅ Missing values filled")

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    st.success(f"✅ Categorical features encoded. Final shape: {X.shape}")

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    st.success("✅ Features scaled")

    # ===============================
    # 🔢 Split + Train
    # ===============================
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    st.sidebar.header("🛠️ Model Hyperparameters")
    penalty = st.sidebar.selectbox("Penalty", ["l1", "l2"])
    solver = st.sidebar.selectbox("Solver", ["liblinear", "saga", "lbfgs", "newton-cg", "sag"])
    C = st.sidebar.slider("Regularization Strength (C)", 0.01, 10.0, 1.0, 0.1)
    max_iter = st.sidebar.slider("Max Iterations", 100, 1000, 300, 50)

    # Fit model
    model = LogisticRegression(penalty=penalty, solver=solver, C=C, max_iter=max_iter)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ===============================
    # 📊 Evaluation
    # ===============================
    st.subheader("📈 Model Evaluation")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")

    # Confusion Matrix
    st.subheader("🧮 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    fig_cm, ax_cm = plt.subplots()
    disp.plot(ax=ax_cm)
    st.pyplot(fig_cm)

    tn, fp, fn, tp = cm.ravel()
    st.write(f"**True Positives:** {tp}, **True Negatives:** {tn}, **False Positives:** {fp}, **False Negatives:** {fn}")

    # ROC & AUC
    st.subheader("🚦 ROC Curve & AUC")
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend()
    st.pyplot(fig_roc)
    st.success(f"✅ ROC AUC Score: {roc_auc:.4f}")
# -------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
