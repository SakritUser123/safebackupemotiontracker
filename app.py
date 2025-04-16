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


if selected_tab == 'Multi Emotion AI':
    # Load model and vectorizer
    with open('OnlineLogReg.pkl', 'rb') as f:
        multi_loaded_model = pickle.load(f)
    with open('OnlineVector.pkl', 'rb') as file:
        multi_vectorizer = pickle.load(file)
    
    # Initialize training data (using pickle files or empty data if it's the first time)
    try:
        with open('x_train.pkl', 'rb') as f:
            x_train = pickle.load(f)
        with open('y_train.pkl', 'rb') as f:
            y_train = pickle.load(f)
    except FileNotFoundError:
        x_train = None  # This should be defined earlier in your script
        y_train = pd.Series()
    
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

        st.session_state.larger_messages.append({"
