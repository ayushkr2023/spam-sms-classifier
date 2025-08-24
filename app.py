import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize stemmer
ps = PorterStemmer()

# Cache nltk downloads so they don’t run every time
@st.cache_resource
def download_nltk_resources():
    for resource in ["punkt", "stopwords"]:
        try:
            if resource == "punkt":
                nltk.data.find("tokenizers/punkt")
            else:
                nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download(resource)

download_nltk_resources()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("✍️ Enter the message:")

if st.button('🔍 Predict'):
    # Preprocess
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    # Output
    if result == 1:
        st.error("🚨 This message is **Spam**!")
    else:
        st.success("✅ This message is **Not Spam**!")
