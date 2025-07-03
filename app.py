import streamlit as st
import tensorflow as tf
import numpy as np
import re
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === Fungsi untuk Membersihkan Teks ===
# Fungsi ini harus sama persis dengan yang di training
def clean_text(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# === Load Model dan Tokenizer ===
# Gunakan cache untuk mempercepat loading model pada interaksi berikutnya
@st.cache_resource
def load_assets():
    """Memuat model dan tokenizer dari file."""
    try:
        model = tf.keras.models.load_model("model_lstm_sentimen.h5")
        with open("tokenizer.pickle", "rb") as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        st.error(f"Gagal memuat aset model atau tokenizer. Pastikan file 'model_lstm_sentimen.h5' dan 'tokenizer.pickle' ada.\nError: {e}")
        return None, None

model, tokenizer = load_assets()

# === Antarmuka Aplikasi Streamlit ===
st.set_page_config(page_title="Analisis Sentimen Review Netflix", page_icon="🎬", layout="centered")
st.title("🎬 Analisis Sentimen Review Netflix")
st.write(
    "Masukkan sebuah review aplikasi Netflix dalam bahasa Inggris, "
    "dan model akan memprediksi sentimennya (Positif, Negatif, atau Netral)."
)

# Text area untuk input pengguna
user_input = st.text_area("Tulis review Anda di sini:", height=150, placeholder="Contoh: This app is amazing! I can watch my favorite shows everywhere.")

# Tombol untuk prediksi
if st.button("Analisis Sentimen", use_container_width=True):
    if model and tokenizer and user_input:
        # 1. Pra-pemrosesan input
        cleaned_input = clean_text(user_input)
        
        # 2. Tokenisasi dan Padding
        sequence = tokenizer.texts_to_sequences([cleaned_input])
        padded_sequence = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
        
        # 3. Prediksi
        with st.spinner('Menganalisis...'):
            prediction = model.predict(padded_sequence)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
        
        # 4. Mapping hasil ke label sentimen
        # Pastikan urutannya sama dengan saat training: {'positive': 0, 'negative': 1, 'neutral': 2}
        class_names = ["Positif 😄", "Negatif 😠", "Netral 😐"]
        predicted_sentiment = class_names[predicted_class_index]
        confidence = prediction[0][predicted_class_index]

        # Tampilkan hasil
        if predicted_class_index == 0: # Positif
            st.success(f"**Hasil Prediksi: {predicted_sentiment}** (Keyakinan: {confidence:.2%})")
        elif predicted_class_index == 1: # Negatif
            st.error(f"**Hasil Prediksi: {predicted_sentiment}** (Keyakinan: {confidence:.2%})")
        else: # Netral
            st.warning(f"**Hasil Prediksi: {predicted_sentiment}** (Keyakinan: {confidence:.2%})")

        st.write("---")
        st.write("**Detail Proses:**")
        st.write(f"**Input Awal:** `{user_input}`")
        st.write(f"**Teks Setelah Dibersihkan:** `{cleaned_input}`")

    elif not user_input:
        st.warning("Mohon masukkan teks review terlebih dahulu.")

st.markdown("---")
st.markdown("Dibuat dengan [Streamlit](https://streamlit.io) & [TensorFlow/Keras](https://www.tensorflow.org/).")