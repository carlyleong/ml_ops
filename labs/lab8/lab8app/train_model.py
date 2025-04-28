import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature

# Load Wine dataset
wine = load_wine(as_frame=True)
X = wine.data
y = wine.target

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)  # different random seed

# Train a simpler model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")

# Log with MLflow
signature = infer_signature(X_train, model.predict(X_train))
input_example = X_train.iloc[:1]

with mlflow.start_run() as run:
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.sklearn.log_model(model, artifact_path="model", signature=signature, input_example=input_example)
    print(f"Run ID: {run.info.run_id}")



