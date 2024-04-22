import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score 

# Function to train the regression model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = 1 - (1 - r2) * ((X_test.shape[0] - 1) / (X_test.shape[0] - len(X_test.columns) - 1))
    # adjusted_r2 = adjusted_r2_score(y_test, y_pred, X_test.shape[1])
    return mse, r2 , adjusted_r2

# Main function
def main():
    # Title of the application
    st.title('Regression Model Application')
    st.subheader('Upload the Preprocessed and Standardaize Data')
    # Upload CSV data file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file)

        # Display the raw data
        st.subheader('Raw Data')
        st.write(data)

        target_column = st.sidebar.selectbox("Select target column", data.columns)

        # Check if target column exists in the dataset
        if target_column not in data.columns:
            st.error("Target column not found in the dataset.")
            return

        # Split data into features and target variable
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = train_model(X_train, y_train)

        # Evaluate the model
        mse, r2, adjusted_r2  = evaluate_model(model, X_test, y_test)

        # Display evaluation metrics
        st.sidebar.subheader('Model Evaluation')
        st.sidebar.write('Mean Squared Error:', mse)
        st.sidebar.write('R^2 Score:', r2)
        st.sidebar.write('Adjusted R-Square Score:',adjusted_r2)

        

# Run the main function
if __name__ == '__main__':
    main()
