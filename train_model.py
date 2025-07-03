import re
import pandas as pd
import numpy as np
import pickle # Import pickle

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional


# === 1. Load & Preprocess Data dengan Pandas ===
# Untuk deployment, lebih mudah menggunakan Pandas daripada PySpark
print("Membaca dataset...")
df = pd.read_csv("netflix_reviews.csv")

# Drop duplikat & handle missing values
df.drop_duplicates(inplace=True)
df.dropna(subset=['content', 'score'], inplace=True)

# === 2. Fungsi untuk Membersihkan Teks & Membuat Sentimen ===
def clean_text(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)  # Hapus tag HTML
    text = re.sub(r'http\S+', '', text)  # Hapus URL
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hapus karakter non-alfabet
    text = re.sub(r'\s+', ' ', text).strip()  # Hapus spasi berlebih
    return text

def score_to_sentiment(score):
    try:
        r = int(score)
        if r >= 4:
            return "positive"
        elif r == 3:
            return "neutral"
        else: # score 1 dan 2
            return "negative"
    except (ValueError, TypeError):
        return None # Akan di-drop nanti

print("Membersihkan teks dan membuat sentimen...")
df['sentiment'] = df['score'].apply(score_to_sentiment)
df.dropna(subset=['sentiment'], inplace=True) # Hapus baris dengan sentimen null

df['clean_content'] = df['content'].apply(clean_text)

# === 3. Label Encoding ===
# Membuat mapping dari sentimen ke angka
sentiment_labels = {'positive': 0, 'negative': 1, 'neutral': 2}
df['label'] = df['sentiment'].map(sentiment_labels)
df.dropna(subset=['label'], inplace=True)
df['label'] = df['label'].astype(int)

# === 4. Tokenisasi dan Padding ===
print("Melakukan tokenisasi teks...")
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean_content'])

sequences = tokenizer.texts_to_sequences(df['clean_content'])
padded_sequences = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')

# Siapkan label
labels = df['label'].values

# === 5. Simpan Tokenizer! (Langkah Krusial) ===
print("Menyimpan tokenizer ke 'tokenizer.pickle'...")
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# === 6. Split Data & Training Model ===
print("Membagi data training dan test...")
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42
)

# === 6. Split Data & Training Model ===
# ...
print("Membangun dan melatih model LSTM...")
model = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    Bidirectional(LSTM(64, return_sequences=False)), # <-- MENJADI SEPERTI INI
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=15, # Anda bisa sesuaikan jumlah epoch
    validation_data=(X_test, y_test),
    verbose=2
)

# === 7. Simpan Model ===
print("Menyimpan model ke 'model_lstm_sentimen.h5'...")
model.save("model_lstm_sentimen.h5")

print("\nTraining selesai! Aset 'model_lstm_sentimen.h5' dan 'tokenizer.pickle' telah dibuat.")