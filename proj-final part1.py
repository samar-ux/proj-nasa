# استيراد المكتبات اللازمة
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer

# تحميل مجموعة البيانات
df = pd.read_excel("C:/Users/Lenovo/Desktop/archive1/nearest-earth-objects_1910-2024_.xlsx")

# عرض أولى الصفوف من مجموعة البيانات
print(df.head())

# التحقق من القيم المفقودة
print(df.isnull().sum())

# التعامل مع القيم المفقودة باستخدام SimpleImputer
imputer = SimpleImputer(strategy='mean')
df[['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max']] = imputer.fit_transform(df[['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max']])

# التحقق من أنواع البيانات
print(df.dtypes)

# تحويل 'is_hazardous' إلى ثنائي (0 و 1)
df['is_hazardous'] = df['is_hazardous'].astype(int)

# التحليل الاستكشافي: توزيع absolute_magnitude
plt.figure(figsize=(10, 6))
sns.histplot(df['absolute_magnitude'], kde=True, bins=30)
plt.title('توزيع absolute_magnitude')
plt.show()

# اختيار الميزات
features = ['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity', 'miss_distance']
X = df[features]
y = df['is_hazardous']

# ترميز المتغير الفئوي 'orbiting_body'
encoder = LabelEncoder()
df['orbiting_body'] = encoder.fit_transform(df['orbiting_body'])

# التعامل مع عدم التوازن في الفئات باستخدام SMOTE
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

# التطبيع/التوحيد
scaler = StandardScaler()
X_res = scaler.fit_transform(X_res)

# تقسيم البيانات إلى مجموعات تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# تدريب النموذج وتقييمه
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVC': SVC(probability=True)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"{name}:")
    print(f"الدقة: {precision}")
    print(f"الاستدعاء: {recall}")
    print(f"درجة F1: {f1}")
    print(f"AUC-ROC: {roc_auc}")
    print()

# اختيار أفضل نموذج
best_model = RandomForestClassifier()
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# رسم منحنى ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label='منحنى ROC')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('معدل الإيجابيات الكاذبة')
plt.ylabel('معدل الإيجابيات الحقيقية')
plt.title('منحنى ROC')
plt.legend()
plt.show()