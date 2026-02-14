import streamlit as st
import numpy as np
import pickle

class LinearRegression:
    def __init__(self, learning_rate, iterations):
        self.coeff = None
        self.intercept = None
        self.lr = learning_rate
        self.epochs = iterations
        self.losses = []

    def predict(self, X):
        return np.dot(X, self.coeff) + self.intercept

# Set up the web page
st.set_page_config(page_title="Placement Predictor")
st.title("🎓 Student Placement Predictor")
st.write("Enter student details to predict placement probability.")


@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Create the user interface for inputs
col1, col2 = st.columns(2)

with col1:
    internships = st.number_input("Number of Internships", min_value=0, max_value=10, value=0)
    cgpa = st.slider("CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)

with col2:
    backlogs = st.selectbox("History of Backlogs", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

if st.button("Predict Placement Status"):
    user_input = np.array([[internships, cgpa, backlogs]])
    
    # Get prediction
    prediction = model.predict(user_input)
    score = prediction[0]
    binary_result = 1 if score >= 0.5 else 0
    
    if binary_result == 1:
        st.success(f"Result: Likely to be Placed! (Score: {score:.2f})")
    else:
        st.error(f"Result: Not Likely to be Placed. (Score: {score:.2f})")
