from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn

# Define your Wine feature input
class InputData(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

# Load model from MLflow
MODEL_URI = "runs:/265ac7a3aac444bf963a8dfac7bc8a13/model"  # <<< paste YOUR Run ID here!
model = mlflow.sklearn.load_model(MODEL_URI)

# Create FastAPI app
app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    features = [[
        data.alcohol,
        data.malic_acid,
        data.ash,
        data.alcalinity_of_ash,
        data.magnesium,
        data.total_phenols,
        data.flavanoids,
        data.nonflavanoid_phenols,
        data.proanthocyanins,
        data.color_intensity,
        data.hue,
        data.od280_od315_of_diluted_wines,
        data.proline
    ]]
    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}

