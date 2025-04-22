import pickle
import streamlit as st
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and vectorizer
with open('SVMNewpkl', 'rb') as f:  # Make sure this matches the saved filename
    model = pickle.load(f)
with open('SVMVectorNew.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Class labels for training
classes = ['joy', 'sad', 'fear', 'suprise', 'anger', 'love']

# Session state for persistent UI elements
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'predicted_emotion' not in st.session_state:
    st.session_state.predicted_emotion = ""

st.title("💬 Emotion Prediction & Online Model Update")

# Text input
user_input = st.text_input("✏️ Enter a sentence:", value=st.session_state.user_input)

# Submit button to make prediction
if st.button("🔍 Predict Emotion"):
    if user_input.strip():
        X_new = vectorizer.transform([user_input])
        predicted = model.predict(X_new)
        st.session_state.predicted_emotion = predicted[0]
        st.success(f"Predicted Emotion: **{predicted[0]}**")

# Dropdown to update label
label = st.selectbox("✅ Confirm or correct the emotion label:", classes)

# Update model button
if st.button("📈 Update Model"):
    X_new = vectorizer.transform([user_input])

    # Prediction before update
    before = model.predict(X_new)[0]

    # First-time setup for partial_fit
    if not hasattr(model, 'classes_'):
        model.partial_fit(X_new, [label], classes=classes)
    else:
        model.partial_fit(X_new, [label])

    # Prediction after update
    after = model.predict(X_new)[0]

    # Save the updated model
    with open('SVMNew.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('SVMVectorNew.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    # Show change
    st.info(f"🔄 Model Prediction Updated\n**Before:** {before}\n**After:** {after}")
    st.session_state.predicted_emotion = ""  # Clear old prediction
    predicted[0] = after
# Save user input in session state
st.session_state.user_input = user_input

# Display prediction if available
if st.session_state.predicted_emotion:
    st.write(f"**Current Predicted Emotion:** {st.session_state.predicted_emotion}")
