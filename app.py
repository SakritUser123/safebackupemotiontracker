import streamlit as st
st.set_page_config(page_title="Emotion Analyzer AI", page_icon="😃")
import pickle

tabs = ["Multi Emotion AI", "EmotionAI"]
selected_tab = st.sidebar.radio("Select A Model", tabs)

# Initialize session states for chat histories
if "smaller_messages" not in st.session_state:
    st.session_state.smaller_messages = []

if "larger_messages" not in st.session_state:
    st.session_state.larger_messages = []


    
    


if selected_tab == 'Multi Emotion AI':
    import pickle
    import numpy as np
    import pandas as pd
    from scipy.sparse import vstack
    import streamlit as st
    
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
    
        st.session_state.larger_messages.append({"role": "user", "content": user_input})
    
        if user_input.strip():
            # Transform user input
            user_input_list = [user_input]
            X = multi_vectorizer.transform(user_input_list)
    
            # Predict with the existing model
            pred = multi_loaded_model.predict(X)
            explain = '0 is for sadness , 1 is for joy, 2 is for love , 3 is for anger , 4 is for fear, 5 is for surprise'
            probabilities = multi_loaded_model.predict_proba(X)[0]
            
            label_to_text = {0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'}
    
            # Show prediction results to the user
            with st.chat_message("assistant"):
                st.markdown(explain)
                st.markdown(f"Prediction: {pred[0]}")
                for i in range(len(probabilities)):
                    emotion = label_to_text[i]
                    percent = round(probabilities[i] * 100, 2)
                    st.markdown(f"- **{emotion.capitalize()}**: {percent}%")
    
            # Get correct label from the user
            corr = st.chat_input("Enter the correct label:")
            if corr:
                # Append the new data to the training set
                X_new = multi_vectorizer.transform([user_input])
                if x_train is not None:
                    x_train = vstack([x_train, X_new])  # Keep x_train sparse
                else:
                    x_train = X_new  # First entry
                
                y_train = pd.concat([y_train, pd.Series([corr])], ignore_index=True)
    
                # Retrain the model with the updated data
                multi_loaded_model.fit(x_train, y_train)
    
                # Save the updated model, vectorizer, and training data
                with open('OnlineLogReg.pkl', 'wb') as f:
                    pickle.dump(multi_loaded_model, f)
                with open('OnlineVector.pkl', 'wb') as file:
                    pickle.dump(multi_vectorizer, file)
                with open('x_train.pkl', 'wb') as f:
                    pickle.dump(x_train, f)
                with open('y_train.pkl', 'wb') as f:
                    pickle.dump(y_train, f)
    
                st.session_state.larger_messages.append({"role": "assistant", "content": f"Model updated with label: {corr}"})
                st.write("Model updated with new data!")
