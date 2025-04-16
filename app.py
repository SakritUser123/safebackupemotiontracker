import streamlit as st
import pickle
import os

# Mapping
label_to_num = {'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}
num_to_label = {v: k for k, v in label_to_num.items()}

# Load model & vectorizer into session state
if "model" not in st.session_state:
    if os.path.exists("SVMLogReg (3).pkl") and os.path.exists("SVMVector (3).pkl"):
        with open("SVMLogReg (3).pkl", "rb") as f:
            st.session_state.model = pickle.load(f)
        with open("SVMVector (3).pkl", "rb") as f:
            st.session_state.vectorizer = pickle.load(f)
    else:
        st.error("Model or vectorizer file not found.")
        st.stop()

model = st.session_state.model
vectorizer = st.session_state.vectorizer

# UI
st.title("🎯 Real-time Emotion Classifier")
user_input = st.text_input("Enter your text:")

if user_input:
    X = vectorizer.transform([user_input])
    pred = model.predict(X)[0]
    st.write("Predicted emotion: )
    st.write(pred)

    correct_label = st.text_input("Correct label (optional):", "")
    if correct_label:
        correct_label = correct_label.lower()
        if correct_label in label_to_num:
            y = label_to_num[correct_label]
            model.partial_fit(X, [y])  # Safe now — it knows all classes
            with open("SVMLogReg (3).pkl", "wb") as f:
                pickle.dump(model, f)
            st.success(f"Model updated with label: {correct_label}")
        else:
            st.warning("Label must be: joy, sadness, love, anger, fear, surprise")

