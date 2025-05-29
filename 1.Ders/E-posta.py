import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1) Veri yükle
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# 2) Özellikler ve etiketler
#    - train_df içinde hem id hem target var
X = train_df.drop(columns=["id", "target"])
y = train_df["target"]

# 3) Eğitim/Doğrulama bölünmesi
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4) Modeli oluştur ve eğit
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5) Doğrulama setinde değerlendirme
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))

# 6) Test seti üzerinde tahmin
X_test = test_df.drop(columns=["id"])
test_preds = model.predict(X_test)

# 7) Submission dosyası oluştur
submission = pd.DataFrame({
    "id": test_df["id"],
    "target": test_preds
})
submission.to_csv("submission.csv", index=False)
print("Tahminler 'submission.csv' olarak kaydedildi.")
