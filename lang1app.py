
import streamlit as st
import pickle

# Load the model and CountVectorizer from the file
with open("nlp_model.pkl", "rb") as model_file:
    cv, model = pickle.load(model_file)

# Streamlit app interface
st.title("Language Prediction App")
st.write("Enter a text snippet to predict its language")

user_input = st.text_area("Enter Text:")

if st.button("Predict Language"):
    if user_input:
        data = cv.transform([user_input]).toarray()
        output = model.predict(data)
        st.write(f"The predicted language is: {output[0]}")
    else:
        st.write("Please enter some text for prediction.")
