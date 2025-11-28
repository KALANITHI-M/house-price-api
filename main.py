from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import gradio as gr
import plotly.graph_objs as go

# ---------------- FASTAPI + MODEL ----------------

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

def compare_scenarios(medInc_a, houseAge_a, aveRooms_a, aveBedrms_a,
                      population_a, aveOccup_a, latitude_a, longitude_a,
                      medInc_b, houseAge_b, aveRooms_b, aveBedrms_b,
                      population_b, aveOccup_b, latitude_b, longitude_b):
    """Compare two different sets of features and return both prices + bar chart."""
    X_a = [[medInc_a, houseAge_a, aveRooms_a, aveBedrms_a,
            population_a, aveOccup_a, latitude_a, longitude_a]]
    X_b = [[medInc_b, houseAge_b, aveRooms_b, aveBedrms_b,
            population_b, aveOccup_b, latitude_b, longitude_b]]

    y_a = float(model.predict(X_a)[0])
    y_b = float(model.predict(X_b)[0])

    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Scenario A", "Scenario B"], y=[y_a, y_b]))
    fig.update_layout(
        title="Scenario Comparison: A vs B",
        yaxis_title="Predicted Price (USD)",
    )

    return y_a, y_b, fig

# ---------- GRADIO UI ----------

with gr.Blocks(title="House Price Predictor") as demo:
    gr.Markdown(
        """
    # 🏠 House Price Predictor

    Explore how different house and neighborhood features affect the predicted price.
    Use the tabs below for **single prediction** or **scenario comparison**.
    """
    )

    with gr.Tabs():
        # ------- TAB 1: Single Prediction -------
        with gr.Tab("Single Prediction"):
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

        # ------- TAB 2: Scenario Comparison -------
        with gr.Tab("Scenario Comparison"):
            gr.Markdown("### 🆚 Compare two different scenarios side by side")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🟦 Scenario A")
                    medInc_a = gr.Slider(0, 15, value=5.0, step=0.1, label="MedInc A")
                    houseAge_a = gr.Slider(1, 60, value=20.0, step=1, label="HouseAge A")
                    aveRooms_a = gr.Slider(1, 15, value=5.0, step=0.1, label="AveRooms A")
                    aveBedrms_a = gr.Slider(0, 5, value=1.0, step=0.05, label="AveBedrms A")
                    population_a = gr.Slider(1, 5000, value=500.0, step=10, label="Population A")
                    aveOccup_a = gr.Slider(0.5, 10, value=3.0, step=0.05, label="AveOccup A")
                    latitude_a = gr.Slider(32, 42, value=36.0, step=0.01, label="Latitude A")
                    longitude_a = gr.Slider(-125, -114, value=-120.0, step=0.01, label="Longitude A")

                with gr.Column():
                    gr.Markdown("#### 🟩 Scenario B")
                    medInc_b = gr.Slider(0, 15, value=10.0, step=0.1, label="MedInc B")
                    houseAge_b = gr.Slider(1, 60, value=40.0, step=1, label="HouseAge B")
                    aveRooms_b = gr.Slider(1, 15, value=7.0, step=0.1, label="AveRooms B")
                    aveBedrms_b = gr.Slider(0, 5, value=1.2, step=0.05, label="AveBedrms B")
                    population_b = gr.Slider(1, 5000, value=300.0, step=10, label="Population B")
                    aveOccup_b = gr.Slider(0.5, 10, value=2.5, step=0.05, label="AveOccup B")
                    latitude_b = gr.Slider(32, 42, value=38.0, step=0.01, label="Latitude B")
                    longitude_b = gr.Slider(-125, -114, value=-121.0, step=0.01, label="Longitude B")

            compare_btn = gr.Button("🔍 Compare Scenarios", variant="primary")

            price_a = gr.Number(label="Scenario A Price (USD)", precision=2)
            price_b = gr.Number(label="Scenario B Price (USD)", precision=2)
            compare_plot = gr.Plot(label="Scenario A vs Scenario B")

            compare_btn.click(
                fn=compare_scenarios,
                inputs=[
                    medInc_a, houseAge_a, aveRooms_a, aveBedrms_a,
                    population_a, aveOccup_a, latitude_a, longitude_a,
                    medInc_b, houseAge_b, aveRooms_b, aveBedrms_b,
                    population_b, aveOccup_b, latitude_b, longitude_b,
                ],
                outputs=[price_a, price_b, compare_plot],
            )

# mount Gradio
app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
