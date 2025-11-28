from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import gradio as gr
import plotly.graph_objs as go

app = FastAPI()
model = joblib.load("house_model.pkl")

class Input(BaseModel):
    data: Optional[list] = [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]

@app.post("/predict")
def predict(input: Input = Input()):
    pred = model.predict([input.data])
    return {"prediction": float(pred[0])}

# ---------- FUNCTIONS ----------

def predict_and_plot(medInc, houseAge, aveRooms, aveBedrms,
                     population, aveOccup, latitude, longitude):
    # current point prediction
    X_curr = [[medInc, houseAge, aveRooms, aveBedrms,
               population, aveOccup, latitude, longitude]]
    y_curr = model.predict(X_curr)[0]

    # build curve: vary MedInc, keep others fixed
    xs = [i / 2 for i in range(0, 31)]  # 0.0 → 15.0 step 0.5
    X_curve = [[x, houseAge, aveRooms, aveBedrms,
                population, aveOccup, latitude, longitude] for x in xs]
    ys = model.predict(X_curve)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Price vs MedInc"))
    fig.add_vline(x=medInc, line_dash="dash", name="Current MedInc")
    fig.update_layout(
        xaxis_title="MedInc (Median Income in 10k$)",
        yaxis_title="Predicted Price (USD)",
        title="Effect of Median Income on Predicted Price",
    )

    return float(y_curr), fig

def load_example():
    return (
        8.3252,
        41.0,
        6.98,
        1.02,
        322.0,
        2.55,
        37.88,
        -122.23,
    )

def clear_inputs():
    return 0, 0, 0, 0, 0, 0, 0, 0

# ---------- GRADIO UI ----------

with gr.Blocks(title="House Price Predictor") as demo:
    gr.Markdown(
        """
    # 🏠 House Price Predictor

    Move the sliders to change the house/neighborhood features.  
    The chart shows how the predicted price changes with **Median Income (MedInc)**.
    """
    )

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### 🔢 Input Features")

            medInc = gr.Slider(0, 15, value=8.3252, step=0.1, label="MedInc (Median Income in 10k$)")
            houseAge = gr.Slider(1, 60, value=41.0, step=1, label="HouseAge (years)")
            aveRooms = gr.Slider(1, 15, value=6.98, step=0.1, label="AveRooms (avg rooms)")
            aveBedrms = gr.Slider(0, 5, value=1.02, step=0.05, label="AveBedrms (avg bedrooms)")
            population = gr.Slider(1, 5000, value=322.0, step=10, label="Population")
            aveOccup = gr.Slider(0.5, 10, value=2.55, step=0.05, label="AveOccup (avg occupants)")
            latitude = gr.Slider(32, 42, value=37.88, step=0.01, label="Latitude")
            longitude = gr.Slider(-125, -114, value=-122.23, step=0.01, label="Longitude")

            with gr.Row():
                predict_btn = gr.Button("🔮 Predict & Show Graph", variant="primary")
                example_btn = gr.Button("📌 Load Sample Values")
                clear_btn = gr.Button("🧹 Clear")

        with gr.Column(scale=2):
            gr.Markdown("### 📊 Output")

            output_price = gr.Number(
                label="Predicted Median House Value (USD)", value=0, precision=2
            )
            price_plot = gr.Plot(label="Price vs Median Income")

            gr.Markdown(
                """
                **How to read the graph:**
                - The line shows predicted price for different values of **MedInc**  
                  (other sliders stay fixed).
                - The dashed vertical line is the **current MedInc** from the slider.
                """
            )

    predict_btn.click(
        fn=predict_and_plot,
        inputs=[medInc, houseAge, aveRooms, aveBedrms, population, aveOccup, latitude, longitude],
        outputs=[output_price, price_plot],
    )

    example_btn.click(
        fn=load_example,
        inputs=[],
        outputs=[medInc, houseAge, aveRooms, aveBedrms, population, aveOccup, latitude, longitude],
    )

    clear_btn.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[medInc, houseAge, aveRooms, aveBedrms, population, aveOccup, latitude, longitude],
    )

# mount Gradio
app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
