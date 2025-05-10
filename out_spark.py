import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# ⏱️ Start timing
start_time = time.time()

# Step 1: Load dataset
df = pd.read_csv("pdc_dataset_with_target.csv")
df = df.drop_duplicates()

# Step 2: Fill missing values
columns_to_fill = ['feature_1', 'feature_2', 'feature_4', 'feature_7']
for col in columns_to_fill:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# Step 3: Encode categorical columns
le_5 = LabelEncoder()
le_3 = LabelEncoder()
df['feature_5_encoded'] = le_5.fit_transform(df['feature_5'].astype(str))
df['feature_3_encoded'] = le_3.fit_transform(df['feature_3'].astype(str))

# Step 4: Log transform
df['feature_4'] = np.log1p(df['feature_4'].clip(lower=0))
df['feature_7'] = np.log1p(df['feature_7'].clip(lower=0))

df['feature_4'] = df['feature_4'].fillna(df['feature_4'].median())

# Step 5: Remove outliers
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

for col in ['feature_1', 'feature_4', 'feature_7']:
    df = remove_outliers(df, col)

# Step 6: Feature scaling
feature_cols = ['feature_1', 'feature_2', 'feature_4', 'feature_6', 'feature_7', 'feature_3_encoded', 'feature_5_encoded']
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# Step 7: Prepare data
X = df[feature_cols]
y = df['target']

# Step 8: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 9: Use RandomForestClassifier with parallelism
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 10: Evaluation
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ F1 Score: {f1:.4f}")
print("🔹 Confusion Matrix:\n", conf_matrix)
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# ⏱️ End timing
end_time = time.time()
print(f"⏱️ Time taken: {end_time - start_time:.2f} seconds")

