import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.dummy import DummyClassifier
import time

def run_knn_c(data_path, target_column, dataset_name, k_values=[1, 2, 3, 5, 10, 25, 50, 100], seed=1387):
    df = pd.read_csv(data_path)
    df = df.dropna()

    selected_features = ["Age", "Genetic_Risk", "Air_Pollution", "Alcohol_Use", "Smoking", "Obesity_Level"]
    X = df[selected_features]

    if target_column == "Target_Severity_Score":
        df["Severity_Binned"] = pd.cut(
            df["Target_Severity_Score"],
            bins=[0, 3.5, 6.5, np.inf],
            labels=[0, 1, 2]
        )
        y = df["Severity_Binned"].astype(int)
    else:
        y = df[target_column]

    print("Binned severity distribution:")
    print(df["Severity_Binned"].value_counts())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=seed, stratify=y
    )

    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    print(f"Dummy Classifier Accuracy: {dummy.score(X_test, y_test):.4f}")

    test_accuracies = []
    training_times = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        start_time = time.time()
        accs = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
        duration = time.time() - start_time

        training_times.append(duration)
        test_accuracies.append(np.mean(accs))
        print(f"k={k}: Cross-Val Accuracy = {np.mean(accs):.4f}, Time = {duration:.4f} sec")

    # Plot validation curve
    plt.figure()
    plt.plot(k_values, test_accuracies, marker='o')
    plt.title(f'kNN Model Complexity ({dataset_name})')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Cross-Validation Accuracy')
    plt.grid(True)
    plt.savefig(f'images/knn_model_complexity_{dataset_name.lower().replace(" ", "_")}.png')
    plt.show()

    # Learning curve
    best_k = k_values[np.argmax(test_accuracies)]
    knn = KNeighborsClassifier(n_neighbors=best_k)

    train_sizes, train_scores, test_scores = learning_curve(
        knn, X_scaled, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 5), shuffle=True, random_state=seed
    )

    plt.figure()
    plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
    plt.plot(train_sizes, test_scores.mean(axis=1), label='Validation')
    plt.title(f'kNN Learning Curve (k={best_k}) ({dataset_name})')
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'images/knn_learning_curve_{dataset_name.lower().replace(" ", "_")}.png')
    plt.show()

    # Confusion Matrix
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train, y_train)
    y_pred = best_knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix (k={best_k}) - {dataset_name}')
    plt.savefig(f'images/knn_confusion_matrix_{dataset_name.lower().replace(" ", "_")}.png')
    plt.show()


if __name__ == "__main__":
    run_knn_c(
        "data/global_cancer_patients_2015_2024.csv",
        target_column="Target_Severity_Score",
        dataset_name="Severity_Classification"
    )