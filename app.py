import pickle
import streamlit as st
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
with open('SVMNewpkl', 'rb') as f:
    model = pickle.load(f)
with open('SVMVectorNew.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Define emotion labels
classes = ['joy', 'sad', 'fear', 'surprise', 'anger', 'love']

# Track state
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'predicted_emotion' not in st.session_state:
    st.session_state.predicted_emotion = ""
if 'first_time' not in st.session_state:
    st.session_state.first_time = True  # Important!

st.title("💬 Emotion Predictor & Online Trainer")

with st.form(key="predict_form"):
    user_input = st.text_input("🔤 Enter a sentence:", value=st.session_state.user_input)
    submit = st.form_submit_button("🎯 Predict")

    if submit:
        if user_input.lower() == 'stop':
            st.info("Exiting.")
        elif user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            X_new = vectorizer.transform([user_input])
            predicted = model.predict(X_new)
            st.session_state.predicted_emotion = predicted[0]
            st.session_state.user_input = user_input

if st.session_state.predicted_emotion:
    st.success(f"🎉 Predicted Emotion: **{st.session_state.predicted_emotion}**")

    label = st.selectbox("✅ Correct label (to update model):", classes)

    if st.button("📈 Update Model"):
        X_new = vectorizer.transform([st.session_state.user_input])

        if st.session_state.first_time:
            model.partial_fit(X_new, [label], classes=classes)  # important on first update
            st.session_state.first_time = False
        else:
            model.partial_fit(X_new, [label])

        with open('SVMNew.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('SVMVectorNew.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)

        st.success("✅ Model updated and learning!")

