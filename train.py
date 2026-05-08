import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss

def train_model():
    print("📈 Loading processed data...")
    df = pd.read_pickle("data/processed/processed_data.pkl")

    # 1. Define Parameters
    max_prediction_length = 21 
    max_encoder_length = 60
    training_cutoff = df["time_idx"].max() - max_prediction_length

    # 2. Create Dataset
    context = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="sales",
        group_ids=["id"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["item_id", "store_id", "cat_id"],
        time_varying_known_reals=["time_idx", "sell_price"],
        time_varying_known_categoricals=["day_of_week", "month"],
        time_varying_unknown_reals=["sales"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    validation = TimeSeriesDataSet.from_dataset(context, df, predict=True, stop_randomization=True)

    # 3. DataLoaders
    train_dataloader = context.to_dataloader(train=True, batch_size=32, num_workers=0, persistent_workers=False)
    val_dataloader = validation.to_dataloader(train=False, batch_size=32, num_workers=0, persistent_workers=False)

    print(f"🤖 Initializing TFT Model...")

    # 4. Setup the Model
    tft = TemporalFusionTransformer.from_dataset(
        context,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=4,
        dropout=0.1,
        loss=QuantileLoss(),
    )

    # 5. Modern Trainer Setup
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto", 
        devices="auto",
        callbacks=[EarlyStopping(monitor="val_loss", patience=3)],
    )

    print("🚀 Training starting...")
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    # 6. Save
    trainer.save_checkpoint("models/demand_model.ckpt")
    print("🎉 Model saved to models/demand_model.ckpt")

if __name__ == "__main__":
    train_model()