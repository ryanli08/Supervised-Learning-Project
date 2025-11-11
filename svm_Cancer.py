import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import time

def run_svm_cancer(data_path, target_column, dataset_name, kernel_types=["linear", "rbf"], seed=42):
    df = pd.read_csv(data_path)
    df = df.dropna()

    # Select features and target
    features = ["Age", "Genetic_Risk", "Air_Pollution", "Alcohol_Use", "Smoking", "Obesity_Level"]
    X = df[features]

    # Target processing
    if target_column == "Cancer_Stage":
        df["Cancer_Stage_Binary"] = df["Cancer_Stage"].apply(lambda x: 1 if x in ["Stage III", "Stage IV"] else 0)
        y = df["Cancer_Stage_Binary"]
    elif target_column == "Target_Severity_Score":
        df["Severity_Binned"] = pd.cut(
            df["Target_Severity_Score"],
            bins=[0, 3.5, 6.5, np.inf],
            labels=[0, 1, 2]
        )
        y = df["Severity_Binned"].astype(int)
        print("Binned severity distribution:")
        print(df["Severity_Binned"].value_counts())
    else:
        y = df[target_column]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=seed, stratify=y
    )

    # Dummy baseline
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    print(f"Dummy Accuracy: {dummy.score(X_test, y_test):.4f}")

    for kernel in kernel_types:
        print(f"\nTraining SVM with kernel = '{kernel}'")
        svm = SVC(kernel=kernel, random_state=seed)

        # Cross-validation and timing
        start_time = time.time()
        accs = cross_val_score(svm, X_scaled, y, cv=5, scoring='accuracy')
        duration = time.time() - start_time
        print(f"SVM ({kernel}) Cross-Val Accuracy: {np.mean(accs):.4f}, Time = {duration:.4f} sec")

        # Learning curve
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        train_sizes, train_scores, test_scores = learning_curve(
            svm, X_scaled, y, cv=cv, scoring="accuracy",
            train_sizes=np.linspace(0.1, 1.0, 5)
        )
        plt.figure()
        plt.plot(train_sizes, train_scores.mean(axis=1), label="Train")
        plt.plot(train_sizes, test_scores.mean(axis=1), label="Validation")
        plt.title(f"SVM Learning Curve (kernel={kernel}) ({dataset_name})")
        plt.xlabel("Training Size")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"images/svm_learning_curve_{dataset_name.lower().replace(' ', '_')}_{kernel}.png")
        plt.show()

        # Fit and plot confusion matrix
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f'Confusion Matrix (kernel={kernel}) - {dataset_name}')
        plt.savefig(f'images/svm_confusion_matrix_{dataset_name.lower().replace(" ", "_")}_{kernel}.png')
        plt.show()


if __name__ == "__main__":
    run_svm_cancer(
        "data/global_cancer_patients_2015_2024.csv",
        target_column="Target_Severity_Score",
        dataset_name="Severity_Classification"
    )
