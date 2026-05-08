import pandas as pd
import matplotlib.pyplot as plt
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

def run_prediction():
    print("🔮 Loading model and data for prediction...")
    
    # 1. Load data and model
    df = pd.read_pickle("data/processed/processed_data.pkl")
    model = TemporalFusionTransformer.load_from_checkpoint("models/demand_model.ckpt")

    # 2. Re-create the dataset parameters (must match training)
    max_prediction_length = 21 
    max_encoder_length = 60
    training_cutoff = df["time_idx"].max() - max_prediction_length

    # 3. Predict on the last 21 days
    # We take the data and ask the model to 'guess' the final 21 days
    raw_predictions = model.predict(df, mode="raw", return_x=True)

    # 4. Plot the results for the first item in the batch
    print("📊 Generating forecast graph...")
    model.plot_prediction(raw_predictions.x, raw_predictions.output, idx=0, add_loss_to_title=True)
    
    # Save the graph so you can see it
    plt.savefig("forecast_result.png")
    print("✅ Forecast graph saved as 'forecast_result.png' in your main folder!")
    plt.show()

if __name__ == "__main__":
    run_prediction()