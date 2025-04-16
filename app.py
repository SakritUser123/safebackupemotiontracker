import streamlit as st
import pickle
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Labels for emotions
label_to_num = {'joy': 0, 'sad': 1, 'fear': 2, 'surprise': 3, 'anger': 4, 'love': 5}
num_to_label = {v: k for k, v in label_to_num.items()}

# Load or initialize model and vectorizer
try:
    with open('SVMLogReg.pkl (3)', 'rb') as f:
        model = pickle.load(f)
    with open('SVMVector.pkl (3)', 'rb') as f:
        vectorizer = pickle.load(f)
    first_time = False
    st.write("Model and vectorizer loaded successfully.")
except:
    # If the model is not found, create a new one
    model = SGDClassifier(loss='log')
    vectorizer = TfidfVectorizer()
    first_time = True
    st.write("Initialized new model and vectorizer.")

# Streamlit UI components
st.title("Emotion Prediction and Model Update")

# Display previous chat messages for larger emotions
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display the chat interface with previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.text_input("Enter a sentence to predict emotion:")

if user_input:
    # Predict emotion
    X_new = vectorizer.transform([user_input])
    prediction = model.predict(X_new)[0]
    

    # Display prediction
    st.write(prediction)

    # Update model with correct label (manual input)
    correct_label = st.selectbox("Enter the correct label if the prediction is wrong:", 
                                 ['joy', 'sad', 'fear', 'surprise', 'anger', 'love'])

    if correct_label:
        # Update model with new data (online learning)
        correct_num = label_to_num[correct_label]
        model.partial_fit(X_new, [correct_num], classes=list(label_to_num.values()))
        st.write("Model updated with the new label!")

        # Save the updated model and vectorizer
        with open('SVMLogReg.pkl (3)', 'wb') as f:
            pickle.dump(model, f)
        with open('SVMVector.pkl (3)', 'wb') as f:
            pickle.dump(vectorizer, f)

        # Store the chat message
        st.session_state.messages.append({"role": "assistant", "content": f"Emotion updated to: {correct_label}"})
        
    # Store user input and prediction
    st.session_state.messages.append({"role": "user", "content": user_input})

st.markdown("---")
st.write("You can enter a sentence, receive a prediction, and manually update the model if the prediction is wrong. This helps improve the model over time.")
