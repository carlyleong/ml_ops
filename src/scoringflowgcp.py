from metaflow import FlowSpec, step, Parameter, JSONType, catch, timeout, retry, conda_base, resources, kubernetes
import mlflow
import mlflow.sklearn

@conda_base(
    python="3.9.16",
    libraries={
        "scikit-learn": "1.2.2",
        "mlflow": "2.2.2",
        "cloudpickle": "2.2.1",
        "pandas": "1.5.3"
    }
)
class ScoringFlowGCP(FlowSpec):
    vector = Parameter("vector", type=JSONType, help="Feature vector for prediction")

    @catch(var="load_fail")
    @timeout(seconds=300)
    @retry(times=2)
    @resources(cpu=1, memory=2048)
    @kubernetes
    @step
    def start(self):
        import numpy as np

        mlflow.set_tracking_uri("http://mlflow:5000")  # Internal service name in K8s

  # K8s-internal MLflow service
        model_name = "metaflow-wine-model"

        model_uri = "models:/metaflow-wine-model/latest"
        self.model = mlflow.sklearn.load_model(model_uri)


        self.vector_np = np.array(self.vector).reshape(1, -1)

        print("Model loaded from MLflow.")
        print("Input vector:", self.vector_np.tolist())
        self.next(self.predict)

    @retry(times=2)
    @resources(cpu=1, memory=2048)
    @kubernetes
    @step
    def predict(self):
        prediction = self.model.predict(self.vector_np)
        print("Predicted class:", prediction[0])
        self.prediction = prediction[0]  # Store for use in end step
        self.next(self.end)

    @step
    def end(self):
        print("Scoring flow complete.")
        print("Final prediction:", self.prediction)

if __name__ == "__main__":
    ScoringFlowGCP()



