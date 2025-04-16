import streamlit as st
import pickle
import numpy as np
import os

# Define consistent label maps
label_to_num = {'sad': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}
num_to_label = {v: k for k, v in label_to_num.items()}
all_labels = list(label_to_num.values())

# Session state to store chat history
if "larger_messages" not in st.session_state:
    st.session_state.larger_messages = []

# Load model and vectorizer
model_path = "SVMLogReg.pkl"
vectorizer_path = "SVMVector.pkl"

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
else:
    st.error("Model or vectorizer not found.")
    st.stop()

# UI
st.title("💬 Multi Emotion Analyzer AI")
st.markdown("Enter a message and let the AI detect the emotion.")
st.markdown("**Labels**: sad, joy, love, anger, fear, surprise")

# Show history
for msg in st.session_state.larger_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
user_input = st.chat_input("Enter your text here...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.larger_messages.append({"role": "user", "content": user_input})

    # Transform input and predict
    X = vectorizer.transform([user_input])
    pred = model.predict(X)
    pred_label = num_to_label[int(pred[0])]

    with st.chat_message("assistant"):
        st.write(f"Predicted emotion: **{pred_label}**")

    # Ask for true label
    correct_label = st.text_input("Enter the correct label (to help me learn):", key=user_input)

    if correct_label and correct_label.lower() in label_to_num:
        correct_num = label_to_num[correct_label.lower()]
        model.partial_fit(X, [correct_num], classes=all_labels)
        st.session_state.larger_messages.append({
            "role": "assistant",
            "content": f"✅ Model updated with label: **{correct_label}**"
        })

        # Save updated model
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        st.success("Model updated and saved!")
    elif correct_label:
        st.error("❌ Invalid label. Use one of: sad, joy, love, anger, fear, surprise.")
