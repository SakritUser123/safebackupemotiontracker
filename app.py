import pickle
import streamlit as st
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize model and vectorizer from pickle

with open('SVMLogReg (3).pkl ', 'rb') as f:
    model = pickle.load(f)
with open('SVMVector (3).pkl', 'rb') as f:
    vectorizer = pickle.load(f)
first_time = False

# Streamlit UI components
st.title("Emotion Prediction and Model Update")

# This session state will be used to keep track of user inputs
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Input for text (enter sentence to predict)
user_input = st.text_input("Enter a sentence to predict emotion (or 'stop' to quit):", value=st.session_state.user_input)

if user_input.lower() == 'stop':
    st.write("Exiting the loop.")
else:
    if user_input:
        # Transform the input using the vectorizer
        X_new = vectorizer.transform([user_input])

        # Predict the emotion
        ans = model.predict(X_new)
        st.write(f"Predicted Emotion: {ans[0]}")

        # Label input from the user
        label = st.selectbox("Enter label (joy, sad, fear, surprise, anger, love):", 
                             ['joy', 'sad', 'fear', 'surprise', 'anger', 'love'])

        # Update model with the label entered by the user
        correct_num = label  # Store the user-provided label
        X_new = vectorizer.transform([user_input])  # Don't refit vectorizer, just transform

        model.partial_fit(X_new, [correct_num])  # Update the model with new data

        # Save the updated model and vectorizer
        with open('SVMLogReg (3).pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('SVMVector (3).pkl', 'wb') as f:
            pickle.dump(vectorizer, f)

        st.write("Model updated!")

    # Save the current user input in the session state to persist across reruns
    st.session_state.user_input = user_input
