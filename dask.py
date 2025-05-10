import time
import dask.dataframe as dd
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# ⏱️ Start total processing timer
start_time = time.time()

# Step 1: Load dataset using Dask
df = dd.read_csv("pdc_dataset_with_target.csv")
df = df.drop_duplicates()

# Step 2: Fill missing values (compute medians first)
columns_to_fill = ['feature_1', 'feature_2', 'feature_4', 'feature_7']
medians = {col: df[col].median().compute() for col in columns_to_fill}
for col in columns_to_fill:
    df[col] = df[col].fillna(medians[col])

# Step 3: Log transform
df['feature_4'] = np.log1p(df['feature_4'].clip(lower=0))
df['feature_7'] = np.log1p(df['feature_7'].clip(lower=0))


median_val = df['feature_4'].median()
df['feature_4'] = df['feature_4'].fillna(median_val)


# Step 4: Encode categorical variables manually (with map)
# Step 3: Encode categorical variables in Dask with meta
df['feature_5'] = df['feature_5'].map({'Yes': 1, 'No': 0}, meta=('feature_5', 'int64'))
df['feature_3'] = df['feature_3'].map({'A': 0, 'B': 1, 'C': 2}, meta=('feature_3', 'int64'))



# Step 5: Remove outliers (define helper)
def remove_outliers(d, col):
    Q1 = d[col].quantile(0.25).compute()
    Q3 = d[col].quantile(0.75).compute()
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return d[(d[col] >= lower) & (d[col] <= upper)]

outlier_cols = ['feature_1', 'feature_4', 'feature_7']
for col in outlier_cols:
    df = remove_outliers(df, col)

# Step 6: Convert to Pandas for scikit-learn
df = df.compute()

# Step 7: Scale features
scaler = StandardScaler()
columns_to_scale = ['feature_1', 'feature_2', 'feature_4', 'feature_6', 'feature_7']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Step 8: Split for ML
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Train RandomForest with parallelism
model = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_estimators=100,
    n_jobs=-1  # 🔸 Parallel processing (all cores)
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 10: Evaluate
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("🔹 Accuracy:", accuracy_score(y_test, y_pred))
print("🔹 F1 Score:", f1_score(y_test, y_pred))

# ⏱️ End timer
end_time = time.time()
print(f"\n⏱️ Total processing time (Dask/Parallel RF): {end_time - start_time:.2f} seconds")

