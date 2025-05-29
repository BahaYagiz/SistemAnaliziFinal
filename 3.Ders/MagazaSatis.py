import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 📥 Verileri yükle
train = pd.read_csv("train.csv", parse_dates=['date'])
test = pd.read_csv("test.csv", parse_dates=['date'])
stores = pd.read_csv("stores.csv")
oil = pd.read_csv("oil.csv", parse_dates=['date'])
holidays = pd.read_csv("holidays_events.csv", parse_dates=['date'])

# 🧹 Temizleme - Gereksiz tatilleri çıkar
holidays = holidays[holidays.transferred == False][['date', 'type']]

# 🔗 Verileri birleştir
train = train.merge(holidays, on='date', how='left')
test = test.merge(holidays, on='date', how='left')

# 🎯 Özellik oluşturma fonksiyonu
def create_features(df):
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    # Hataları önlemek için 'type' kontrolü
    if 'type' in df.columns:
        df['holiday'] = df['type'].notnull().astype(int)
    else:
        df['holiday'] = 0
    return df

# Özellikleri oluştur
train = create_features(train)
test = create_features(test)

# 🔍 Hedef ve özellikler
features = ['store_nbr', 'day', 'weekday', 'month', 'year', 'is_weekend', 'holiday']
target = 'sales'

# 🎓 Model eğitimi
model = LGBMRegressor()
model.fit(train[features], train[target])

# 📊 Tahmin yap
predictions = model.predict(test[features])
test['predicted_sales'] = predictions

# 💾 Tahminleri kaydet
test[['id', 'predicted_sales']].to_csv("tahmin_sonuclari.csv", index=False)

# 🖼️ İsteğe bağlı: Tahmin görselleştirme
plt.figure(figsize=(12, 5))
plt.plot(predictions[:100], label='Tahmin')
plt.title("İlk 100 Satış Tahmini")
plt.legend()
plt.show()
