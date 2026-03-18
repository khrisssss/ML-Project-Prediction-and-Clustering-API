from fastapi import FastAPI
import joblib

app = FastAPI(
    title="ML Prediction API",
    description="Predict Y from (A, B) and determine the cluster class of (A, B, Y).",
    version="1.0.0",
)

# Load models
reg_model     = joblib.load('best_model.pkl')
cluster_model = joblib.load('kmeans_model.pkl')
scaler        = joblib.load('scaler.pkl')

@app.post("/predict", tags=["Prediction"])    
def predict(a: float, b: float):
    """ Predict the value of Y from (A, B) """
    y = reg_model.predict([[a, b]])[0]
    return {"A": a, "B": b, "Y": float(y)}

@app.get("/classe", tags=["Clustering"])
def get_classe(a: float, b: float, y: float):
    """ Determine the cluster class of (A, B, Y) """
    ab_scaled = scaler.transform([[a, b]])
    cluster = cluster_model.predict(ab_scaled)[0]
    return {"A": a, "B": b, "Y": y, "classe": int(cluster)}

