"""
🎓 SISTEM PREDIKSI NILAI MATEMATIKA SISWA BERBASIS FAKTOR SOSIAL

Arsitektur Sistem:
User
 ↓
Streamlit UI
 ↓
Model Machine Learning (Linear Regression)
 ↓
Prediksi Nilai

Input Fitur:
- Gender
- Race/Ethnicity  
- Parental Level of Education
- Lunch Type
- Test Preparation Course

Output:
- Prediksi Nilai Matematika (0-100)
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np

print("=" * 60)
print("🎓 TRAINING MODEL PREDIKSI NILAI MATEMATIKA SISWA")
print("=" * 60)

# 1. Load Dataset
print("\n📊 Loading dataset...")
df = pd.read_csv("StudentsPerformance.csv")

print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nKolom yang tersedia:")
for col in df.columns:
    print(f"  - {col}")

# 2. Explorasi Data
print("\n📈 Statistik Dataset:")
print(df.describe())

print("\n📋 Info Dataset:")
print(df.info())

# 3. Encoding Fitur Kategorikal
print("\n🔄 Encoding fitur kategorikal...")
df_encoded = pd.get_dummies(df, drop_first=True)

# 4. Tentukan X & y (hanya prediksi math score)
X = df_encoded.drop(["math score", "reading score", "writing score"], axis=1)
y = df_encoded["math score"]

print(f"\n✅ Fitur yang digunakan untuk prediksi ({len(X.columns)} fitur):")
for col in X.columns:
    print(f"  - {col}")

# 5. Split Data (80% training, 20% testing)
print("\n✂️ Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✅ Training set: {X_train.shape[0]} samples")
print(f"✅ Testing set: {X_test.shape[0]} samples")

# 6. Training Model Linear Regression
print("\n🤖 Training Linear Regression Model...")
model = LinearRegression()
model.fit(X_train, y_train)

print("✅ Model berhasil di-training!")

# 7. Evaluasi Model
print("\n📊 EVALUASI MODEL:")
print("-" * 60)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE  (Mean Absolute Error)     : {mae:.2f}")
print(f"RMSE (Root Mean Squared Error) : {rmse:.2f}")
print(f"R²   (Coefficient of Determination): {r2:.4f}")

print(f"\n💡 Interpretasi:")
print(f"   - Model rata-rata meleset {mae:.2f} poin dari nilai sebenarnya")
print(f"   - Model menjelaskan {r2*100:.2f}% variasi dalam nilai matematika")

# 8. Interpretasi Koefisien (Fitur paling berpengaruh)
print("\n🔍 FITUR PALING BERPENGARUH:")
print("-" * 60)
coef = pd.Series(model.coef_, index=X.columns)
top_features = coef.sort_values(ascending=False).head(5)

print("Top 5 Fitur Positif (meningkatkan nilai):")
for feat, val in top_features.items():
    print(f"  {feat:40s} : +{val:.2f}")

bottom_features = coef.sort_values(ascending=True).head(5)
print("\nTop 5 Fitur Negatif (menurunkan nilai):")
for feat, val in bottom_features.items():
    print(f"  {feat:40s} : {val:.2f}")

# 9. Simpan Model
print("\n💾 Menyimpan model...")
joblib.dump(model, "model.pkl")

# Simpan juga nama kolom untuk validasi di app.py
joblib.dump(X.columns.tolist(), "feature_names.pkl")

print("✅ Model disimpan ke 'model.pkl'")
print("✅ Feature names disimpan ke 'feature_names.pkl'")

print("\n" + "=" * 60)
print("✨ TRAINING SELESAI! Model siap digunakan di Streamlit")
print("=" * 60)
print("\n📌 Jalankan aplikasi dengan: streamlit run app.py")
