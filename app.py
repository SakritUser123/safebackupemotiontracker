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


    
    

if selected_tab == 'EmotionAI':
    with open('LogisticRegModel.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    with open('WorkVector.pkl', 'rb') as file:
        vectorizer = pickle.load(file)

    st.title("💬 Emotion Analyzer AI")

    # Display previous chat messages for Smaller Emotion
    for msg in st.session_state.smaller_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Enter your text here...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.smaller_messages.append({"role": "user", "content": user_input})

        if user_input.strip():
            user_input_list = [user_input]
            predictions = loaded_model.predict_proba(vectorizer.transform([user_input]))[0][1]

            emotion = "The Decimal Given Is How Sure The Model Thinks The statement is positive if the decimal is greater than 0.50 then, it is more likely to be positive!"
            emotion_response = predictions 
            emoji = ''
            res = ''
            if emotion_response > 0.50:
                emoji = '😊'
                res = 'Wow! That is great to hear. You can listen to this song to match your emotion.'
            if emotion_response < 0.50:
                emoji = '😔'
                res = 'Oh no! That’s sad to hear. You can feel better by listening to this song that matches your emotion!'
            if emotion_response == 0.50:
                emoji = '🤔'

            with st.chat_message("assistant"):
                st.markdown(emotion)
                st.markdown(emotion_response)
                st.markdown(emoji)
                st.markdown(res)

            st.session_state.smaller_messages.append({"role": "assistant", "content": f"{emotion_response} {emoji} {res}"})

if selected_tab == 'Multi Emotion AI':
    with open('MultiLogRegModel.pkl', 'rb') as f:
        multi_loaded_model = pickle.load(f)
    with open('MultiWorkVector.pkl', 'rb') as file:
        multi_vectorizer = pickle.load(file)

    st.title("💬 Multi Emotion Analyzer AI")
    st.link_button("💻 Pay $3 on Venmo 🤖😊", "https://venmo.com/SakritUser123?txn=pay&amount=3")
    st.write("You can donate to this website; it will help out a lot!")
    st.markdown("""
    ---
    Contact: [veerendrasakthi.prabhurajan@gmail.com]  
    GitHub: [The repository for this website!](https://github.com/SakritUser123/emotiontrackerai)
    """)
    # Display previous chat messages for Larger Emotion
    for msg in st.session_state.larger_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Enter your text here...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.larger_messages.append({"role": "user", "content": user_input})

        if user_input.strip():
            user_input_list = [user_input]
            predictions = multi_loaded_model.predict(multi_vectorizer.transform([user_input]))
            explain = '0 is for sadness , 1 is for joy,2 is for love , 3 is for anger , 4 is for fear and 5 is for surprise'
            probabilities = multi_loaded_model.predict_proba(multi_vectorizer.transform([user_input]))[0]
            label_to_text = {
            0:'sadness',
            1:'joy',
            2:'love',
            3:'anger',
            4:'fear',
            5:'surprise'
            }
            
                
            with st.chat_message("assistant"):
                st.markdown(explain)
                st.markdown(predictions)
                for i in range(len(probabilities)):
                    emotion = label_to_text[i]
                    percent = round(probabilities[i] * 100,2)
                    st.markdown(f"- **{emotion.capitalize()}**: {percent}%")
                
            st.session_state.larger_messages.append({"role": "assistant", "content": predictions})
