import streamlit as st
import re
import pandas as pd
import pickle

key_norm = pd.read_csv('Dataset/key_norm (1).csv')
stopwords_ind = pd.read_csv('Dataset/stopwords_indonesian.csv')

with open('Pickled/stemmer.pkl', 'rb') as stemmer_file:
    stemmer = pickle.load(stemmer_file)

with open('Pickled/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('Pickled/chi2_features.pkl', 'rb') as chi2_features_file:
    chi2_features = pickle.load(chi2_features_file)
    
with open('Pickled/model_lr.pkl', 'rb') as model_lr_file:
    model_lr = pickle.load(model_lr_file)

def casefolding(text):
    if isinstance(text, str):  # Pastikan hanya string yang diproses
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Menghapus karakter non-alphabet
    return text

def text_normalize(text):
    text = ' '.join([key_norm[key_norm['singkat'] == word]['hasil'].values[0] if (key_norm['singkat'] == word).any() else word for word in text.split()])
    text = str.lower(text)
    return text

def remove_stop_words(text):
    clean_words = []
    text = text.split()
    for word in text:
        if word not in stopwords_ind['stopwords'].to_list():
            clean_words.append(word)
    return " ".join(clean_words)

def stemming(text):
    return stemmer.stem(text)

def predict_sentiment(text):
    # casefold
    casefold = casefolding(text)

    # normalize
    textnormalize = text_normalize(casefold)

    # remove stopwords
    removestopwords = remove_stop_words(textnormalize)
    
    # stem words
    stemwords = stemming(removestopwords)

    # vectorize
    vectorized = vectorizer.transform([stemwords]).toarray()

    # feature selection
    X_chi2 = chi2_features.transform(vectorized)

    # predict
    prediction = model_lr.predict(X_chi2)

    return prediction[0]

# Streamlit app
st.title('Sentimen Analisis Ketertarikan Pembelian Mobil Listrik SUZUKI Menggunakan Metode Klasifikasi Logistic Regression')

st.subheader('Model Prediction')

# Text area for user input
user_input = st.text_area("Enter your text here:")

# Submit button
if st.button("Submit", type='primary'):
    # Show spinner while predicting sentiment
    with st.spinner("Predicting sentiment..."):
        # Predict sentiment
        sentiment = predict_sentiment(user_input)
        if sentiment == 'positif':
            # Display the result
            st.success(f"Sentiment: {sentiment.capitalize()}")
        elif sentiment == 'negatif':
            # Display the result
            st.error(f"Sentiment: {sentiment.capitalize()}")
        else:
            # Display the result
            st.write(f"Sentiment: {sentiment.capitalize()}")

st.subheader('Model Performance')
performance_option = st.selectbox(
    "Pilih salah satu opsi:",
    ("---", "Pemilihan Fitur Menggunakan Chi-Square untuk 500 Fitur", "Pemilihan Fitur Menggunakan Chi-Square untuk K-Fitur"),
)
if performance_option == '---':
    pass
elif performance_option == 'Pemilihan Fitur Menggunakan Chi-Square untuk 500 Fitur':
    st.title('Pemilihan fitur Menggunakan Chi-Square untuk 500 Fitur')
    st.subheader('Number of Features')

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.write('Sebelum')
            st.markdown("<h3 style='font-size: 30px;'>2565 Features</h3>", unsafe_allow_html=True)

    with col2:
        with st.container(border=True):
            st.write('Sesudah')
            st.markdown("<h3 style='font-size: 30px;'>500 Features</h3>", unsafe_allow_html=True)

    st.subheader('Confusion Matrix')
    st.image('assets/confm_500.png', use_container_width=True)
    st.subheader('Classification Report')
    st.markdown("""
        ```
     ->              precision    recall  f1-score   support

     negatif       0.82      0.90      0.86        60
     positif       0.90      0.83      0.86        69

    accuracy                           0.86       129
   macro avg       0.86      0.86      0.86       129
weighted avg       0.86      0.86      0.86       129
                
        
""")
elif performance_option == 'Pemilihan Fitur Menggunakan Chi-Square untuk K-Fitur':
    st.title('Pemilihan fitur Menggunakan Chi-Square untuk K-Fitur')

    st.subheader('Eksperimen')
    st.image('assets/eksperimen.png', use_container_width=True)

    st.subheader('Number of Features')
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.write('Sebelum')
            st.markdown("<h3 style='font-size: 30px;'>2565 Features</h3>", unsafe_allow_html=True)

    with col2:
        with st.container(border=True):
            st.write('Sesudah')
            st.markdown("<h3 style='font-size: 30px;'>1999 Features</h3>", unsafe_allow_html=True)

    st.subheader('Confusion Matrix')
    st.image('assets/confm_k.png', use_container_width=True)
    st.subheader('Classification Report')
    st.markdown("""
        ```
     ->              precision    recall  f1-score   support

     negatif       0.98      1.00      0.99        60
     positif       1.00      0.99      0.99        69

    accuracy                           0.99       129
   macro avg       0.99      0.99      0.99       129
weighted avg       0.99      0.99      0.99       129
                
        
""")