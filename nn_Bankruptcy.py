import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.dummy import DummyClassifier
import time

def run_nn_bankruptcy(data_path, target_column="Bankrupt?", dataset_name="Bankruptcy", seed=1387):
    df = pd.read_csv(data_path).dropna()
    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=seed, stratify=y
    )

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_acc = dummy.score(X_test, y_test)

    hidden_layer_configs = [(16,), (32,), (64,), (32, 16), (64, 32)]
    activations = ["relu", "tanh"]
    results = []

    best_acc = 0
    best_config = None
    best_activation = None
    training_times = []

    for act in activations:
        accs = []
        for config in hidden_layer_configs:
            model = MLPClassifier(hidden_layer_sizes=config, activation=act,
                                  max_iter=500, random_state=seed)
            start = time.time()
            model.fit(X_train, y_train)
            duration = time.time() - start

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accs.append(acc)
            training_times.append(duration)
            print(f"{act}, {config}: Test Accuracy = {acc:.4f}, Time = {duration:.2f}s")

            if acc > best_acc:
                best_acc = acc
                best_config = config
                best_activation = act
        results.append((act, accs))

    # Best model learning curve
    best_model = MLPClassifier(hidden_layer_sizes=best_config, activation=best_activation,
                               max_iter=500, random_state=seed)
    print(f"\nGenerating learning curve for best model: {best_activation}, {best_config}")

    train_sizes, train_scores, test_scores = learning_curve(
        best_model, X_scaled, y, cv=5, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 5), shuffle=True, random_state=seed
    )

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
    plt.plot(train_sizes, test_scores.mean(axis=1), label='Validation')
    plt.title(f"NN Learning Curve ({best_activation}, {best_config}) - {dataset_name}")
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/nn_learning_curve_{best_activation}_{str(best_config).replace(', ', '_').replace('(', '').replace(')', '')}_{dataset_name.lower()}.png")
    plt.show()

    # Accuracy by architecture bar chart
    x_labels = [str(cfg) for cfg in hidden_layer_configs]
    x = np.arange(len(hidden_layer_configs))
    width = 0.35

    plt.figure(figsize=(10, 6))
    for i, (act, accs) in enumerate(results):
        plt.bar(x + i * width, accs, width=width, label=f"{act}")

    plt.axhline(dummy_acc, color='red', linestyle='--', label='Dummy Baseline')
    plt.xticks(x + width / 2, x_labels, rotation=45)
    plt.xlabel("Hidden Layer Configuration")
    plt.ylabel("Test Accuracy")
    plt.title("NN Accuracy by Activation & Architecture (Bankruptcy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/nn_activation_comparison_bankruptcy.png")
    plt.show()

    # Confusion matrix for best model
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {dataset_name} ({best_activation}, {best_config})")
    plt.savefig(f"images/nn_confusion_matrix_{dataset_name.lower()}.png")
    plt.show()

if __name__ == "__main__":
    run_nn_bankruptcy("data/bankruptcy_data.csv", target_column="Bankrupt?", dataset_name="Bankruptcy", seed=1387)

