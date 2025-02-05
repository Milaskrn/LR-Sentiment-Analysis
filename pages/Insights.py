import streamlit as st

st.title('Sentimen Analisis Ketertarikan Pembelian Mobil Listrik SUZUKI Menggunakan Metode Klasifikasi Logistic Regression')

st.subheader('Insights')

# insight 1
st.markdown("<h3 style='text-align: center;'>Jumlah Sentimen tiap Channel</h3>", unsafe_allow_html=True)
st.image('assets/channel_dist.png', use_container_width=True)

# insight 2
st.markdown("<h3 style='text-align: center;'>WordCloud Sentimen Positif dan Negatif</h3>", unsafe_allow_html=True)
st.image('assets/wordcloud.png', use_container_width=True)

# insight 3
st.markdown("<h3 style='text-align: center;'>Top 10 Kata Kunci dengan Nilai Rata-rata TF-IDF (Data Training)</h3>", unsafe_allow_html=True)
st.image('assets/train_tfidf.png', use_container_width=True)

# insight 4
st.markdown("<h3 style='text-align: center;'>Top 10 Kata Kunci dengan Nilai Rata-rata TF-IDF (Data Testing)</h3>", unsafe_allow_html=True)
st.image('assets/test_tfidf.png', use_container_width=True)