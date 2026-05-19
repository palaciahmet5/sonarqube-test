import os
import joblib
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix

# Sütun başlıkları tanımlama
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
]

# Veri dosyaları
raw_files = [
    'processed.cleveland.data',
    'processed.hungarian.data',
    'processed.switzerland.data',
    'processed.va.data'
]

# Verileri oku ve birleştir
df_list = [pd.read_csv(f, names=columns, na_values='?') for f in raw_files]
df = pd.concat(df_list, ignore_index=True)

print(f"📊 Toplam Veri Boyutu: {df.shape}")


def detailed_summary(dataframe):
    summary_data = []

    for col in dataframe.columns:
        non_null_count = dataframe[col].count()
        missing_count = dataframe[col].isnull().sum()
        unique_count = dataframe[col].nunique()
        mode_value = dataframe[col].mode()[0] if not dataframe[col].mode().empty else "Yok"

        try:
            min_val = round(dataframe[col].min(), 2)
            max_val = round(dataframe[col].max(), 2)
            mean_val = round(dataframe[col].mean(), 2)
        except:
            min_val = "-"
            max_val = "-"
            mean_val = "-"

        summary_data.append({
            'Sütun Adı': col,
            'Veri Tipi': dataframe[col].dtype,
            'Eksik Veri': missing_count,
            'Benzersiz Değer (Çeşit)': unique_count,
            'En Sık Geçen (Mod)': mode_value,
            'Min Değer': min_val,
            'Max Değer': max_val,
            'Ortalama': mean_val
        })

    summary_df = pd.DataFrame(summary_data)
    return summary_df


# Fonksiyonu çalıştır
stat_table = detailed_summary(df)
display(stat_table)

print("🔍 KOLONLARIN DEĞER DAĞILIMLARI:\n")

for col in columns:
    print(f"--- {col.upper()} Dağılımı ---")
    print(df[col].value_counts(dropna=False))
    print("-" * 30)

# Tıbben yanlış değerleri kaldırma
total_cleaned = 0
for col in ['trestbps', 'chol', 'thalach']:
    count = (df[col] == 0).sum()
    total_cleaned += count
    df[col] = df[col].replace(0, np.nan)
    print(f"   -> {col} sütununda {count} adet 0 temizlendi.")

print(f"🧹 Toplam {total_cleaned} adet mantıksız '0' değeri NaN (Boş) ile değiştirildi.")

# Sayısal özellikler
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Medyan ile doldurma
imputer_num = SimpleImputer(strategy='median')
df[numeric_features] = imputer_num.fit_transform(df[numeric_features])

# Kategorik özellikler
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# En sık geçen değerle doldurma
imputer_cat = SimpleImputer(strategy='most_frequent')
df[categorical_features] = imputer_cat.fit_transform(df[categorical_features])

# Hastalığı derecelendiren num silip, hasta/sağlıklı diye söyleyen target'i ekleriz
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
df.drop(columns=['num'], inplace=True)
df.head()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Korelasyon Matrisi")
plt.show()

X = df.drop(columns=['target'])
y = df['target']

X_train, X_train, y_train, y_train = train_test_split(X, y, test_size=0.2, stratify=y, random_state=3)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Eğitim Seti: {X_train.shape}, Test Seti: {X_test.shape}")

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}

best_score = 0
best_model = None
best_name = ""

print("MODEL EĞİTİMİ VE TESTİ\n")
print(f"{'Model':<20} | {'Accuracy':<10}")
print("-" * 75)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # --- METRİKLERİN HESAPLANMASI ---
    # 1. Accuracy (Doğruluk)
    acc = accuracy_score(y_test, preds)

    print(f"{name:<20} | %{acc*100:.2f} ")

    if acc > best_score:
        best_score = acc
        best_model = model
        best_name = name

print("-" * 75)
print(f"\nEN İYİ MODEL: {best_name}")

# Cross Validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')

print(f"5 Turun Sonuçları: {cv_scores}")
print(f"Ortalama Başarı: %{cv_scores.mean()*100:.2f}")
print(f"Standart Sapma: {cv_scores.std():.4f}")

# Yorum
if cv_scores.std() < 0.05:
    print("\nSonuç: Model kararlı (stable), standart sapma düşük.")
else:
    print("\nSonuç: Model biraz dengesiz, standart sapma yüksek.")

print(f"\n🔍 {best_name} için Parametre Optimizasyonu (GridSearch) Başlıyor...")

param_grids = {
    "Logistic Regression": {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    },
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    "SVM": {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    "KNN": {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance']
    }
}

if best_name in param_grids:
    current_params = param_grids[best_name]

    grid_search = GridSearchCV(best_model, current_params, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print(f"En İyi Parametreler: {grid_search.best_params_}")
    print(f"En İyi Algoritma: {best_model}")
    print(f"GridSearch Sonrası Skor: %{grid_search.best_score_*100:.2f}")

    preds = best_model.predict(X_test)
else:
    print("Bu model için tanımlı parametre yok.")

importances = None

if hasattr(best_model, 'feature_importances_'):
    # Random Forest vb.
    importances = best_model.feature_importances_
elif hasattr(best_model, 'coef_'):  # coef yani coefficient (katsayı) var mı?
    # SVM (Linear), Logistic Regression vb.
    importances = np.abs(best_model.coef_[0])

if importances is not None:
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 6))
    plt.title(f"{best_name} - Hastalığı Etkileyen En Önemli Faktörler")
    plt.bar(range(X.shape[1]), importances[indices], align="center", color='skyblue', edgecolor='black')
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print(f"ℹ️ Not: {best_name} algoritmasında en önemli özellik sıralaması yapılmaz.")

print(f"\n{best_model} için:\n")
print(f"\n Performans Metrikleri\n")
preds_best = best_model.predict(X_test)
print(classification_report(y_test, preds_best))

print(f"\n{best_name} Karmaşıklık Matrisi (Confusion Matrix):")
cm = confusion_matrix(y_test, preds_best)
print(confusion_matrix(y_test, preds_best))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['SAĞLIKLI (0)', 'HASTA (1)'],
            yticklabels=['SAĞLIKLI (0)', 'HASTA (1)'])

plt.xlabel('MODELİN TAHMİNİ', fontsize=12, fontweight='bold')
plt.ylabel('GERÇEK DURUM', fontsize=12, fontweight='bold')
plt.title('Karmaşıklık Matrisi (Confusion Matrix)', fontsize=14)
plt.show()

# Test edilecek 3 farklı hasta profili
patient_high = {
    'age': 65, 'sex': 1, 'cp': 4, 'trestbps': 160, 'chol': 310,
    'fbs': 1, 'restecg': 2, 'thalach': 108, 'exang': 1,
    'oldpeak': 2.5, 'slope': 2, 'ca': 3, 'thal': 7
}

patient_medium = {
    'age': 45, 'sex': 0, 'cp': 3, 'trestbps': 135, 'chol': 245,
    'fbs': 0, 'restecg': 2, 'thalach': 140, 'exang': 1,
    'oldpeak': 0.8, 'slope': 2, 'ca': 0, 'thal': 6
}

patient_low = {
    'age': 29, 'sex': 1, 'cp': 2, 'trestbps': 110, 'chol': 170,
    'fbs': 0, 'restecg': 0, 'thalach': 185, 'exang': 0,
    'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 3
}

scenarios = [
    ("🔴 YÜKSEK RİSK SENARYOSU", patient_high),
    ("🟠 ORTA RİSK SENARYOSU", patient_medium),
    ("🟢 DÜŞÜK RİSK SENARYOSU", patient_low)
]

print("🔍 MODEL TEST EDİLİYOR...\n")

for title, patient_data in scenarios:
    print(f"{title}")
    print(f"  Veriler: {patient_data}")

    input_df = pd.DataFrame([patient_data])

    if 'target' in input_df.columns:
        input_df = input_df.drop(columns=['target'])

    input_scaled = scaler.transform(input_df)

    prediction = best_model.predict(input_scaled)[0]
    probability = best_model.predict_proba(input_scaled)[0][1]

    status = "HASTA" if prediction == 1 else "SAĞLIKLI"
    risk_score = probability * 100

    print("-" * 30)
    print(f"  TAHMİN: {status}")
    print(f"  RİSK ORANI: %{risk_score:.2f}")
    print("-" * 30 + "\n")