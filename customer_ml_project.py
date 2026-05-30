import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

DATA_PATH = "customer_retail csv file.csv"


def generate_sample_dataset(path: str, n_samples: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    countries = ["United Kingdom", "Germany", "France", "Spain", "Netherlands"]
    quantity = np.random.randint(1, 50, size=n_samples)
    unit_price = np.round(np.random.uniform(1.0, 150.0, size=n_samples), 2)
    country = np.random.choice(countries, size=n_samples, p=[0.5, 0.15, 0.15, 0.1, 0.1])
    df = pd.DataFrame({
        "Quantity": quantity,
        "UnitPrice": unit_price,
        "Country": country,
    })
    # Introduce a few missing values for preprocessing demonstration
    missing_idx = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
    df.loc[missing_idx, "UnitPrice"] = np.nan
    missing_idx = np.random.choice(df.index, size=int(n_samples * 0.03), replace=False)
    df.loc[missing_idx, "Country"] = None
    df.to_csv(path, index=False)
    return df


def load_customer_data(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        print(f"Loading dataset from {path}")
        return pd.read_csv(path)

    print(f"Dataset not found at {path}. Generating a sample dataset")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return generate_sample_dataset(path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Handling missing values and preprocessing data...")
    df = df.copy()

    if "Quantity" not in df.columns or "UnitPrice" not in df.columns or "Country" not in df.columns:
        raise ValueError("Dataset must contain Quantity, UnitPrice, and Country columns")

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["Country"] = df["Country"].fillna("Unknown").astype(str)

    df["Quantity"] = df["Quantity"].fillna(df["Quantity"].median())
    df["UnitPrice"] = df["UnitPrice"].fillna(df["UnitPrice"].median())

    if "CustomerType" not in df.columns:
        median_total = (df["Quantity"] * df["UnitPrice"]).median()
        df["CustomerType"] = np.where(
            df["Quantity"] * df["UnitPrice"] >= median_total,
            "HighValue",
            "Regular",
        )

    df["CustomerType"] = df["CustomerType"].fillna("Regular").astype(str)

    label_encoders = {
        "Country": LabelEncoder(),
        "CustomerType": LabelEncoder(),
    }
    df["CountryEncoded"] = label_encoders["Country"].fit_transform(df["Country"])
    df["Target"] = label_encoders["CustomerType"].fit_transform(df["CustomerType"])

    print("Finished preprocessing")
    return df, label_encoders


def plot_customer_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    df["Country"].value_counts().plot(kind="bar", color="skyblue")
    plt.title("Customer Distribution by Country")
    plt.xlabel("Country")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("customer_distribution.png")
    plt.close()
    print("Saved customer distribution plot: customer_distribution.png")


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {"model": model, "accuracy": acc, "confusion_matrix": cm}
        print(f"{name} accuracy: {acc:.4f}")
    return results


def plot_accuracy_comparison(results):
    plt.figure(figsize=(8, 5))
    names = list(results.keys())
    accuracies = [results[name]["accuracy"] for name in names]
    plt.bar(names, accuracies, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.ylim(0, 1)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    for index, value in enumerate(accuracies):
        plt.text(index, value + 0.01, f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig("model_accuracy_comparison.png")
    plt.close()
    print("Saved accuracy comparison plot: model_accuracy_comparison.png")


def plot_confusion_matrices(results):
    plt.figure(figsize=(15, 4))
    for idx, (name, data) in enumerate(results.items(), start=1):
        cm = data["confusion_matrix"]
        plt.subplot(1, 3, idx)
        plt.imshow(cm, cmap="Blues")
        plt.title(name)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks([0, 1])
        plt.yticks([0, 1])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig("confusion_matrices.png")
    plt.close()
    print("Saved confusion matrices plot: confusion_matrices.png")


def main():
    df = load_customer_data(DATA_PATH)
    df, label_encoders = preprocess_data(df)

    plot_customer_distribution(df)

    features = ["Quantity", "UnitPrice", "CountryEncoded"]
    X = df[features]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    results = train_and_evaluate(models, X_train, X_test, y_train, y_test)
    plot_accuracy_comparison(results)
    plot_confusion_matrices(results)

    print("\nSummary")
    for name, data in results.items():
        print(f"- {name}: Accuracy = {data['accuracy']:.4f}")
    print("\nProject complete. Generated plots: customer_distribution.png, model_accuracy_comparison.png, confusion_matrices.png")


if __name__ == "__main__":
    main()
