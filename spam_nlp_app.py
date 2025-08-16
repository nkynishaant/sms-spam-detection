#stream lit UI
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

st.title("SMS Spam Detector")

user1_msg=st.text_area("Enter your message here:",height=120)

if st.button("Check if Spam"):
    ps = PorterStemmer()
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)                   # Remove numbers
        text = re.sub(r'\W', ' ', text)                   # Remove non-words
        text = re.sub(r'\s+', ' ', text)                  # Remove extra spaces
        text = text.strip()
        
        # Tokenize and remove stopwords + stemming
        cleaned_words = []
        for word in text.split():
            if word not in stopwords.words('english'):
                stemmed = ps.stem(word)      # Stemming the word
                cleaned_words.append(stemmed)
        return " ".join(cleaned_words)
    msg_clean=clean_text(user1_msg)
    vector=pickle.load(open("vectorize_model.pkl","rb"))
    msg_vec=vector.transform([msg_clean])
    modelnb=pickle.load(open("model_nb.pkl","rb"))
    result=modelnb.predict(msg_vec)
    st.write("Spam" if result[0]==1 else "Ham (Not Spam)")
