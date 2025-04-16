import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from scipy.sparse import vstack

# Emotion label to numerical conversion
label_to_num = {'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}

# All possible labels (for partial_fit)
all_labels = [0, 1, 2, 3, 4, 5]  # These correspond to sadness, joy, love, anger, fear, surprise

# Load model and vectorizer (SVM model)
try:
    with open('SVMLogReg (3).pkl', 'rb') as f:
        svm_loaded_model = pickle.load(f)

    with open('SVMVector (3).pkl', 'rb') as file:
        svm_vectorizer = pickle.load(file)

    st.write("Model and vectorizer loaded successfully.")

except Exception as e:
    st.write(f"Error loading model or vectorizer: {e}")
    st.stop()  # Stop the app if loading fails

# Display UI components
st.title("💬 Multi Emotion Analyzer AI")
st.link_button("💻 Pay $3 on Venmo 🤖😊", "https://venmo.com/SakritUser123?txn=pay&amount=3")
st.write("You can donate to this website; it will help out a lot!")
st.markdown("""
---
Contact: [veerendrasakthi.prabhurajan@gmail.com]  
GitHub: [The repository for this website!](https://github.com/SakritUser123/emotiontrackerai)
""")

# Display previous chat messages for Larger Emotion
for msg in st.session_state.get('larger_messages', []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Enter your text here...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.larger_messages.append({"role": "user", "content": user_input})

    if user_input.strip():
        # Transform user input
        user_input_list = [user_input]
        X = svm_vectorizer.transform(user_input_list)

        # Predict with the existing model
        pred = svm_loaded_model.predict(X)
        explain = '0 is for sadness, 1 is for joy, 2 is for love, 3 is for anger, 4 is for fear, 5 is for surprise'

        label_to_text = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

        # Show prediction results to the user
        with st.chat_message("assistant"):
            st.markdown(explain)
            st.markdown("Prediction: ")
            st.markdown(f'The emotion you are feeling is: {label_to_text[pred[0]]}')

        # Ask for correct label from the user (using st.text_input)
        correct_label = st.text_input("Enter the correct label (e.g., joy, sadness, etc.):")

        if correct_label:
            # Convert correct label to numerical value
            correct_label_num = label_to_num.get(correct_label.lower())

            if correct_label_num is not None:
                # Update the model with the new data (partial_fit)
                try:
                    X_new = svm_vectorizer.transform([user_input])
                    svm_loaded_model.partial_fit(X_new, [correct_label_num], classes=all_labels)  # Use numerical label and classes

                    st.session_state.larger_messages.append({"role": "assistant", "content": f"Model updated with label: {correct_label}"})
                    st.write("Model updated with new data!")
                except Exception as e:
                    st.write(f"Error updating model: {e}")
            else:
                st.write("Invalid label entered. Please use one of the following: joy, sadness, love, anger, fear, surprise.")
