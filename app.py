import pickle
import streamlit as st
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
with open('SVMNewpkl', 'rb') as f:
    model = pickle.load(f)
with open('SVMVectorNew.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

first_time = False

# Initialize session state
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'predicted_emotion' not in st.session_state:
    st.session_state.predicted_emotion = ""

# App title
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>💬 Emotion Predictor & Trainer</h1>", unsafe_allow_html=True)
st.markdown("### 👇 Enter a sentence and label the emotion if the prediction is wrong.")

# Form UI to avoid multiple button clicks triggering everything
with st.form(key="predict_form"):
    user_input = st.text_input("🔤 Enter a sentence:", value=st.session_state.user_input, key="text_input")

    submit = st.form_submit_button("🎯 Predict")

    if submit:
        if user_input.lower() == 'stop':
            st.info("🚪 Exiting the app. Refresh to restart.")
        elif user_input.strip() == "":
            st.warning("Please enter a sentence.")
        else:
            X_new = vectorizer.transform([user_input])
            predicted = model.predict(X_new)
            st.session_state.predicted_emotion = predicted[0]
            st.session_state.user_input = user_input  # save user input

# Show prediction result if available
if st.session_state.predicted_emotion:
    st.success(f"🎉 Predicted Emotion: **{st.session_state.predicted_emotion}**")

    st.markdown("#### 🛠️ Was that correct? If not, help train the model:")

    label = st.selectbox("✅ Choose correct label:", 
                         ['joy', 'sad', 'fear', 'surprise', 'anger', 'love'], key="label_select")

    if st.button("📈 Update Model with New Label"):
        X_new = vectorizer.transform([st.session_state.user_input])
        model.partial_fit(X_new, [label])

        with open('SVMNew.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('SVMVectorNew.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)

        st.success("✅ Model successfully updated!")
