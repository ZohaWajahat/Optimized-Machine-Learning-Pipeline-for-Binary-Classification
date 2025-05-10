import time
import findspark
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log1p, rand
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import classification_report

# Start timer
start_time = time.time()

# Initialize Spark
findspark.init()
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/usr/local/lib/python3.11/dist-packages/pyspark"

spark = SparkSession.builder.appName("Spark_RF_Balanced").getOrCreate()

# Step 1: Load dataset
df = spark.read.csv("pdc_dataset_with_target.csv", header=True, inferSchema=True).dropDuplicates()

# Step 2: Fill missing values
for col_name in ['feature_1', 'feature_2', 'feature_4', 'feature_7']:
    median_val = df.approxQuantile(col_name, [0.5], 0)[0]
    df = df.fillna({col_name: median_val})

# Step 3: Encode categorical columns
df = StringIndexer(inputCol="feature_3", outputCol="feature_3_encoded").fit(df).transform(df)
df = StringIndexer(inputCol="feature_5", outputCol="feature_5_encoded").fit(df).transform(df)

# Step 4: Log transform
df = df.withColumn("feature_4", log1p(col("feature_4").cast("double")))
df = df.withColumn("feature_7", log1p(col("feature_7").cast("double")))

# Step 5: Remove outliers
def remove_outliers(df, col_name):
    Q1 = df.approxQuantile(col_name, [0.25], 0)[0]
    Q3 = df.approxQuantile(col_name, [0.75], 0)[0]
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return df.filter((col(col_name) >= lower) & (col(col_name) <= upper))

for col_name in ['feature_1', 'feature_4', 'feature_7']:
    df = remove_outliers(df, col_name)

# Step 6: Balance classes (undersample majority class)
counts = df.groupBy("target").count().collect()
minority_count = min([row['count'] for row in counts])
balanced_df = df.filter(col("target") == 0).orderBy(rand()).limit(minority_count) \
    .union(df.filter(col("target") == 1).orderBy(rand()).limit(minority_count))

# Step 7: Assemble and scale features
features = ['feature_1', 'feature_2', 'feature_4', 'feature_6', 'feature_7', 'feature_3_encoded', 'feature_5_encoded']
assembler = VectorAssembler(inputCols=features, outputCol="features")
scaled_df = assembler.transform(balanced_df)

scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(scaled_df)
scaled_df = scaler_model.transform(scaled_df)

# Step 8: Train/test split
train_data, test_data = scaled_df.randomSplit([0.8, 0.2], seed=42)

# Step 9: Random Forest model
rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="target", numTrees=100, seed=42)
model = rf.fit(train_data)
predictions = model.transform(test_data)

# Step 10: Evaluation
evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"✅ Accuracy: {accuracy:.4f}")

# Confusion matrix
print("🔹 Confusion Matrix:")
predictions.groupBy("target", "prediction").count().show()

# Classification report (with sklearn)
pred_pd = predictions.select("target", "prediction").toPandas()
print("📊 Classification Report:")
print(classification_report(pred_pd["target"], pred_pd["prediction"]))

# Timing
print(f"⏱️ Time taken: {time.time() - start_time:.2f} seconds")

# Stop Spark
spark.stop()

