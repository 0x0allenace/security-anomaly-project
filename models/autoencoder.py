import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )

        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def get_default_autoencoder_features():
    return [
        "risk_score",
        "session_duration_min",
        "login_freq_5",
        "failed_attempts_rolling",
        "session_zscore",
        "failed_zscore"
    ]


def prepare_autoencoder_data(df, feature_cols):
    df_model = df.copy()
    df_model = df_model.dropna(subset=feature_cols).copy()

    X = df_model[feature_cols].copy()
    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df_model, X_scaled, scaler


def train_autoencoder(
    X_scaled,
    epochs=50,
    batch_size=64,
    learning_rate=1e-3,
    random_state=42
):
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_scaled.shape[1]
    model = Autoencoder(input_dim)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in dataloader:
            batch_x = batch[0]

            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")

    return model


def compute_reconstruction_error(model, X_scaled):
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        reconstructed = model(X_tensor).numpy()

    reconstruction_error = np.mean((X_scaled - reconstructed) ** 2, axis=1)
    return reconstruction_error


def run_autoencoder(
    df,
    feature_cols=None,
    epochs=50,
    batch_size=64,
    learning_rate=1e-3,
    contamination=0.05,
    random_state=42
):
    """
    Apply Autoencoder-based anomaly detection.

    Parameters:
        df (pd.DataFrame): Input dataframe
        feature_cols (list): Features used for training
        epochs (int): Training epochs
        batch_size (int): Batch size
        learning_rate (float): Optimizer learning rate
        contamination (float): Expected anomaly proportion
        random_state (int): Random seed

    Returns:
        df_model (pd.DataFrame): Dataframe with autoencoder outputs
        model (Autoencoder): Trained autoencoder
        scaler (StandardScaler): Fitted scaler
        threshold (float): Reconstruction error threshold
    """

    if feature_cols is None:
        feature_cols = get_default_autoencoder_features()

    df_model, X_scaled, scaler = prepare_autoencoder_data(df, feature_cols)

    model = train_autoencoder(
        X_scaled=X_scaled,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_state=random_state
    )

    reconstruction_error = compute_reconstruction_error(model, X_scaled)

    # Threshold based on contamination
    threshold = np.quantile(reconstruction_error, 1 - contamination)

    df_model["autoencoder_score"] = reconstruction_error
    df_model["autoencoder_anomaly"] = (
        df_model["autoencoder_score"] > threshold).astype(int)

    return df_model, model, scaler, threshold


def save_autoencoder_results(df, output_path="data/processed/processed_logs_with_autoencoder.csv"):
    df.to_csv(output_path, index=False)
    print(f"Autoencoder results saved to {output_path}")


def print_autoencoder_summary(df):
    print("\nAutoencoder Summary")
    print(df["autoencoder_anomaly"].value_counts())

    if "is_attack" in df.columns:
        print("\nGround Truth vs Autoencoder")
        print(pd.crosstab(df["is_attack"], df["autoencoder_anomaly"]))


def main():
    input_path = "data/processed/processed_logs_with_stats.csv"
    output_path = "data/processed/processed_logs_with_autoencoder.csv"

    df = pd.read_csv(input_path)

    feature_cols = get_default_autoencoder_features()

    df_auto, model, scaler, threshold = run_autoencoder(
        df=df,
        feature_cols=feature_cols,
        epochs=50,
        batch_size=64,
        learning_rate=1e-3,
        contamination=0.05,
        random_state=42
    )

    save_autoencoder_results(df_auto, output_path=output_path)

    print(f"Reconstruction error threshold: {threshold:.6f}")
    print_autoencoder_summary(df_auto)

    print("\nPreview:")
    print(df_auto.head())


if __name__ == "__main__":
    main()
