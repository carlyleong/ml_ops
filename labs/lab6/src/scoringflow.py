from metaflow import FlowSpec, step, Parameter, JSONType
import mlflow
import mlflow.sklearn
import os

class ScoringFlow(FlowSpec):
    """
    A Metaflow pipeline that loads a registered model and performs prediction on input data.
    """

    vector = Parameter("vector", type=JSONType, required=True)

    @step
    def start(self):
        """
        Load the model from MLflow registry.
        """
        os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
        mlflow.set_tracking_uri("http://127.0.0.1:5000")

        model_uri = "models:/metaflow-wine-model/1"  # Or "models:/metaflow-wine-model/Production" if promoted
        self.model = mlflow.sklearn.load_model(model_uri)
        self.next(self.predict)

    @step
    def predict(self):
        """
        Perform prediction.
        """
        self.prediction = self.model.predict([self.vector])[0]
        self.next(self.end)

    @step
    def end(self):
        """
        Output prediction result.
        """
        print(f"Input vector: {self.vector}")
        print(f"Predicted class: {self.prediction}")

if __name__ == "__main__":
    ScoringFlow()


