
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# 1. Dataset load karna
data_folder = Path("data/raw")
csv_files = list(data_folder.glob("*.csv"))

if not csv_files:
    print("No CSV file found in data/raw")
    exit()

file_path = csv_files[0]
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# 2. Target column
target_column = "engagement_rate"


# 3. Features
# Post-publication outcome columns intentionally exclude kiye gaye hain
input_features = [
    "account_type",
    "follower_count",
    "media_type",
    "content_category",
    "traffic_source",
    "has_call_to_action",
    "post_hour",
    "day_of_week",
    "caption_length",
    "hashtags_count"
]


# 4. Required columns check
required_columns = input_features + [target_column]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("Missing columns:", missing_columns)
    exit()


# 5. Missing values remove karna
df = df[required_columns].dropna()

print("Shape after removing missing values:", df.shape)


# 6. Input and target separate karna
X = df[input_features]
y = df[target_column]


# 7. Categorical features
categorical_features = [
    "account_type",
    "media_type",
    "content_category",
    "traffic_source",
    "day_of_week"
]


# 8. Numerical features
numerical_features = [
    "follower_count",
    "has_call_to_action",
    "post_hour",
    "caption_length",
    "hashtags_count"
]


# 9. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTrain-test split completed!")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# 10. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# 11. Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)


# 12. Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)


# 13. Model training
print("\nModel training started...")

pipeline.fit(X_train, y_train)

print("Model training completed!")


# 14. Predictions
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)


# 15. Evaluation function
def evaluate_model(actual, predicted, dataset_name):

    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)

    print("\n" + "=" * 50)
    print(dataset_name)
    print("=" * 50)

    print("MSE:", round(mse, 6))
    print("MAE:", round(mae, 6))
    print("RMSE:", round(rmse, 6))
    print("R²:", round(r2, 6))


# 16. Training evaluation
evaluate_model(
    y_train,
    y_train_pred,
    "TRAINING EVALUATION"
)


# 17. Testing evaluation
evaluate_model(
    y_test,
    y_test_pred,
    "TESTING EVALUATION"
)


print("\nRegression evaluation completed successfully!")