from metaflow import FlowSpec, step, conda_base, kubernetes, resources, retry, timeout, catch

@conda_base(
    python='3.9.16',
    libraries={
        'scikit-learn': '1.2.2',
        'mlflow': '2.9.2',
        'databricks-cli': '0.17.7'
    }
)
class TrainingFlowGCP(FlowSpec):

    @catch(var="train_fail")
    @timeout(seconds=300)
    @retry(times=2)
    @resources(cpu=2, memory=4096)
    @kubernetes
    @step
    def start(self):
        from sklearn import datasets
        from sklearn.model_selection import train_test_split

        X, y = datasets.load_wine(return_X_y=True)
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(X, y, test_size=0.2, random_state=0)
        print("Data loaded successfully")
        self.next(self.train_knn, self.train_svm)

    @retry(times=2)
    @resources(cpu=2, memory=4096)
    @kubernetes
    @step
    def train_knn(self):
        from sklearn.neighbors import KNeighborsClassifier

        self.model = KNeighborsClassifier()
        self.model.fit(self.train_data, self.train_labels)
        self.next(self.choose_model)

    @retry(times=2)
    @resources(cpu=2, memory=4096)
    @kubernetes
    @step
    def train_svm(self):
        from sklearn import svm

        self.model = svm.SVC(kernel='poly')
        self.model.fit(self.train_data, self.train_labels)
        self.next(self.choose_model)

    @retry(times=2)
    @resources(cpu=2, memory=4096)
    @kubernetes
    @step
    def choose_model(self, inputs):
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri("http://mlflow-service.argo.svc.cluster.local:5000")
  # Internal service name in K8s

 # Internal MLflow service on K8s
        mlflow.set_experiment("metaflow-experiment")

        def score(inp):
            return inp.model, inp.model.score(inp.test_data, inp.test_labels)

        self.results = sorted(map(score, inputs), key=lambda x: -x[1])
        self.model = self.results[0][0]
        with mlflow.start_run():
            mlflow.sklearn.log_model(self.model, artifact_path="metaflow_train", registered_model_name="metaflow-wine-model")
        self.next(self.end)

    @step
    def end(self):
        print("Scores:")
        print("\n".join("%s %f" % res for res in self.results))
        print("Best model:", self.model)

if __name__ == '__main__':
    TrainingFlowGCP()









