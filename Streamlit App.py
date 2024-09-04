import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st

#load the updated model to be mode live
model = joblib.load("liveModelVI.pkl")

#load the data to check accuarcy
df = pd.read_csv('mobile_price_range_data.csv')
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=42)

#make predictions for X_test set
y_pred = model.predict(X_test)
#Accuarcy
accuracy = accuracy_score(y_train,y_test)

#page title
st.title("Model Accuracy and Real-Time Prediction")
st.write(f"Model {accuracy}")

#Real time prediction based on user inputs
st.header("Real-Time prediction")
input_data = []
for col in X_test.columns:
    input_value = st.numner_input(f'Input for feature {col}', value='')
    inpiut_data.append(input_value)

#convert input data to dataframe
input_df = pd.DataFrame([input_data],columns=X_test.columns)

#make predictions
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write(f'Prediction: {prediction[0]}')