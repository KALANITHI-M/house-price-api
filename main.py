from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import gradio as gr

# Create FastAPI app
app = FastAPI()

# Load model
model = joblib.load("house_model.pkl")

# --------- FastAPI JSON API ---------

class Input(BaseModel):
    data: Optional[list] = [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]

@app.post("/predict")
def predict(input: Input = Input()):
    pred = model.predict([input.data])
    return {"prediction": float(pred[0])}

# --------- Gradio UI ---------

def gradio_predict(medInc, houseAge, aveRooms, aveBedrms,
                   population, aveOccup, latitude, longitude):
    data = [[medInc, houseAge, aveRooms, aveBedrms,
             population, aveOccup, latitude, longitude]]
    pred = model.predict(data)[0]
    return float(pred)

demo = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Number(value=8.3252, label="MedInc"),
        gr.Number(value=41.0,   label="HouseAge"),
        gr.Number(value=6.98,   label="AveRooms"),
        gr.Number(value=1.02,   label="AveBedrms"),
        gr.Number(value=322.0,  label="Population"),
        gr.Number(value=2.55,   label="AveOccup"),
        gr.Number(value=37.88,  label="Latitude"),
        gr.Number(value=-122.23,label="Longitude"),
    ],
    outputs=gr.Number(label="Predicted Price"),
    title="House Price Predictor",
    submit_btn="Predict",   # optional: change button text from "Flag" to "Predict"
)

# Mount Gradio at /ui
app = gr.mount_gradio_app(app, demo, path="/ui")

# For local running (Render uses: uvicorn main:app ...)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
