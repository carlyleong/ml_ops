from metaflow import FlowSpec, step, Parameter
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import os

# Optional: move feature transformations to a separate module
def transform_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

class TrainingFlow(FlowSpec):
    """
    A Metaflow pipeline that ingests data, applies transformations,
    trains and evaluates two models, and registers the best one with MLflow.
    """

    test_size = Parameter("test_size", default=0.2)
    seed = Parameter("seed", default=42)

    @step
    def start(self):
        """Load and transform the dataset."""
        from sklearn import datasets  # explicitly import for modularity

        X, y = load_wine(return_X_y=True)

        # Simulate using separate data processing module
        # Equivalent to: import dataprocessing; X = dataprocessing.transform(X)
        X = transform_features(X)

        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed
        )
        self.next(self.train_knn, self.train_svm)

    @step
    def train_knn(self):
        """Train KNN model."""
        self.model = KNeighborsClassifier()
        self.model.fit(self.train_data, self.train_labels)
        self.next(self.choose_model)

    @step
    def train_svm(self):
        """Train SVM model."""
        self.model = SVC(kernel="linear", probability=True)
        self.model.fit(self.train_data, self.train_labels)
        self.next(self.choose_model)

    @step
    def choose_model(self, inputs):
        """Choose the best model and register with MLflow."""
        import mlflow
        import mlflow.sklearn

        # Ensure subprocesses connect to local MLflow
        os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
        mlflow.set_tracking_uri("http://127.0.0.1:5000")

        def evaluate(input_obj):
            model = input_obj.model
            score = model.score(input_obj.test_data, input_obj.test_labels)
            return model, score

        self.results = sorted(map(evaluate, inputs), key=lambda x: -x[1])
        self.model = self.results[0][0]
        self.best_score = self.results[0][1]

        with mlflow.start_run():
            mlflow.sklearn.log_model(self.model, "model", registered_model_name="metaflow-wine-model")
            mlflow.log_metric("best_accuracy", self.best_score)

        self.next(self.end)

    @step
    def end(self):
        """Print summary."""
        print(f"Training complete. Best model accuracy: {self.best_score:.4f}")

if __name__ == "__main__":
    TrainingFlow()







