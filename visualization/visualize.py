import os
import matplotlib.pyplot as plt


OUTPUT_DIR = "../outputs/figures"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_distribution(df, column, save=False, filename=None):
    ensure_output_dir()

    plt.figure(figsize=(8, 5))
    plt.hist(df[column], bins=50)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    if save:
        if filename is None:
            filename = f"{column}_distribution.png"
        plt.savefig(os.path.join(OUTPUT_DIR, filename),
                    dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()


def plot_anomalies(df, column, anomalies, save=False, filename=None):
    ensure_output_dir()

    plt.figure(figsize=(10, 5))
    plt.scatter(df.index, df[column], label="Normal", alpha=0.6)
    plt.scatter(anomalies.index,
                anomalies[column], color="red", label="Anomalies", alpha=0.9)

    plt.title(f"Anomalies in {column}")
    plt.xlabel("Index")
    plt.ylabel(column)
    plt.legend()
    plt.grid(True)

    if save:
        if filename is None:
            filename = f"{column}_anomalies.png"
        plt.savefig(os.path.join(OUTPUT_DIR, filename),
                    dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()
