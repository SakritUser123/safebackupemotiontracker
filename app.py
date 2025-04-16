import streamlit as st
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import vstack

# Check the selected tab
tabs = ["Multi Emotion AI", "EmotionAI"]
selected_tab = st.sidebar.radio("Select A Model", tabs)

# Initialize session states for chat histories
if "smaller_messages" not in st.session_state:
    st.session_state.smaller_messages = []

if "larger_messages" not in st.session_state:
    st.session_state.larger_messages = []

# Emotion label to numerical conversion
label_to_num = {'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}
label_to_text = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

if selected_tab == 'Multi Emotion AI':
    # Load model and vectorizer (SVM model)
    try:
        with open('SVMLogReg (3).pkl', 'rb') as f:
            svm_loaded_model = pickle.load(f)
        with open('SVMVector (3).pkl', 'rb') as file:
            svm_vectorizer = pickle.load(file)
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        st.stop()

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
            pred = svm_loaded_model.predict(X)[0]

            explain = '0 = sadness, 1 = joy, 2 = love, 3 = anger, 4 = fear, 5 = surprise'

            # Show prediction results to the user
            with st.chat_message("assistant"):
                st.markdown(explain)
                st.markdown(pred)

        # Ask for correct label from the user
        correct_label = st.text_input("✏️ Enter the correct label (joy, sadness, love, anger, fear, surprise):")

        if correct_label:
            correct_label_num = label_to_num.get(correct_label.lower())

            if correct_label_num is not None:
                # Update the model
                X_new = svm_vectorizer.transform([user_input])
                try:
                    svm_loaded_model.partial_fit(X_new, [correct_label_num])

                    # Save model and vectorizer after update
                    with open("SVMLogReg (3).pkl", "wb") as f:
                        pickle.dump(svm_loaded_model, f)
                    with open("SVMVector (3).pkl", "wb") as f:
                        pickle.dump(svm_vectorizer, f)

                    st.success(f"✅ Model updated and saved with label: {correct_label}")
                    st.session_state.larger_messages.append({
                        "role": "assistant",
                        "content": f"Model updated with label: {correct_label}"
                    })
                except Exception as e:
                    st.error(f"Error updating model: {e}")
            else:
                st.warning("Invalid label. Please use one of: joy, sadness, love, anger, fear, surprise.")
