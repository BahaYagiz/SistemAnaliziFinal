# NLP Disaster Tweets Classification Project

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re

# --- 1. Veriyi Yükle ---
train_df = pd.read_csv('train.csv')

# --- 2. Veriyi Temizle ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)  # URL'leri kaldır
    return text

train_df['text'] = train_df['text'].apply(clean_text)

# --- 3. Özellik Çıkart (TF-IDF) ---
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(train_df['text'])
y = train_df['target']

# --- 4. Eğitim/Test Verilerini Ayır ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. Modeli Eğit ---
model = LogisticRegression()
model.fit(X_train, y_train)

# --- 6. Doğruluk Testi ---
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))

# --- 7. Test Verisi Tahmini ve Submission Dosyası ---
test_df = pd.read_csv('test.csv')
test_df['text'] = test_df['text'].apply(clean_text)
X_test_tfidf = vectorizer.transform(test_df['text'])
test_preds = model.predict(X_test_tfidf)

submission = pd.DataFrame({
    'id': test_df['id'],
    'target': test_preds
})

submission.to_csv('submission.csv', index=False)
print("submission.csv başarıyla oluşturuldu!")
