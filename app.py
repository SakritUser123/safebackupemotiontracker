import streamlit as st
import pickle
import os

# Define model paths
model_path = "SVMLogReg (3).pkl"
vectorizer_path = "SVMVector (3).pkl"

# Define emotion label mappings
label_to_num = {'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}
num_to_label = {v: k for k, v in label_to_num.items()}
all_labels = list(label_to_num.values())

# Load model and vectorizer (only once at the start of the session)
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("Model or vectorizer file not found!")
    st.stop()

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit interface
st.title("💬 Multi Emotion Analyzer AI")
st.markdown("Supports emotions: `sad`, `joy`, `love`, `anger`, `fear`, `surprise`")

# Chat input box
user_input = st.text_input("Enter your sentence:")

if user_input:
    # Transform input text to vector using the vectorizer
    X = vectorizer.transform([user_input])

    # Make prediction using the loaded model
    pred = model.predict(X)[0]
    
    
    st.write(pred)

    # Ask for user input to update the model (optional)
    correct_label = st.text_input("Enter correct label to improve the model (optional):")

    if correct_label:
        correct_label = correct_label.lower().strip()
        
        # Ensure the user input is a valid label
        if correct_label in label_to_num:
            correct_num = label_to_num[correct_label]

            # Update the model with the new data using partial_fit
            model.partial_fit(X, [correct_num], classes=all_labels)

            # Save the updated model back to the file
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            st.success(f"✅ Model updated with new label: {correct_label}")
        else:
            st.error("Invalid label! Please enter one of: sad, joy, love, anger, fear, surprise.")
