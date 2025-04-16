import streamlit as st
import pickle
import os

# Load model and vectorizer (update these filenames if needed)
model_path = "SVMLogReg (3).pkl"
vectorizer_path = "SVMVector (3).pkl"

# Emotion label mappings
label_to_num = {'sad': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}
num_to_label = {v: k for k, v in label_to_num.items()}
all_labels = list(label_to_num.values())

# Load model and vectorizer
with open(model_path, "rb") as f:
    model = pickle.load(f)
with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

st.title("💬 Multi Emotion Analyzer AI")
st.markdown("Enter a message and let the AI detect the emotion.")
st.markdown("**Labels**: sad, joy, love, anger, fear, surprise")

# Chat input
user_input = st.chat_input("Enter your text here...")

if user_input:
    # Predict emotion
    X = vectorizer.transform([user_input])
    pred = model.predict(X)[0]
    

    st.write(pred)

    # Ask user for the correct label
    correct_label = st.text_input("Enter the correct label to update the model (or leave empty to skip):")

    if correct_label:
        correct_label = correct_label.lower().strip()
        if correct_label in label_to_num:
            correct_num = label_to_num[correct_label]
            model.partial_fit(X, [correct_num], classes=all_labels)

            # Save updated model
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            st.success(f"✅ Model updated with new label: {correct_label}")
        else:
            st.error("Invalid label! Please enter one of: sad, joy, love, anger, fear, surprise.")
