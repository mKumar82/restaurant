import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(layout="wide")

scaler = joblib.load("Scaler.pkl")

st.title("Restaurant Rating presiction app")



st.caption("this app helps you predict restaurant rating class")

st.divider()

averagecost = st.number_input("please enter the estimated cost for two",min_value=50,max_value=999999,value=1000,step=200)

tablebooking = st.selectbox("restaurant has table booking?",["yes","no"])

onlinedelivery = st.selectbox("restaurant has online delivery?",["yes","no"])

pricerange = st.selectbox("what is the price range (1 cheapest ,4 most expensive)",[1,2,3,4])

predictbutton = st.button("predict the review")

st.divider()

model = joblib.load("mlmodel.pkl")

bookingstatus = 1 if tablebooking=="yes" else 0
deliverystatus = 1 if onlinedelivery=="yes" else 0

values=[[averagecost,bookingstatus,deliverystatus,pricerange]]
my_X_values=np.array(values)

X = scaler.transform(my_X_values)

if predictbutton:
    st.snow()

    predictions = model.predict(X)
    # st.write(predictions)

    if predictions<2.5:
        st.write("poor")
    elif predictions<3.5:
        st.write("Average")
    elif predictions<4.0:
        st.write("Good")
    elif predictions<4.5:
        st.write("Very Good")
    else :
        st.write("Excellent")
