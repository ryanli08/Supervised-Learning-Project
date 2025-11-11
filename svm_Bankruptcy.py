import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.dummy import DummyClassifier
import time

def run_svm_b(data_path, target_column, dataset_name, kernels=["linear", "rbf"], seed=1387):
    df = pd.read_csv(data_path)
    df = df.dropna()

    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, stratify=y, random_state=seed
    )

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    print(f"Dummy Accuracy: {dummy.score(X_test, y_test):.4f}")

    for kernel in kernels:
        print(f"\nTraining SVM with kernel = '{kernel}'")
        model = SVC(kernel=kernel, random_state=seed)

        start_time = time.time()
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        duration = time.time() - start_time

        mean_acc = np.mean(cv_scores)
        print(f"SVM ({kernel}) CV Accuracy: {mean_acc:.4f}, Time = {duration:.4f} sec")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f'Confusion Matrix - SVM ({kernel}) ({dataset_name})')
        plt.savefig(f'images/svm_confusion_matrix_{kernel}_{dataset_name.lower()}.png', dpi=300)
        plt.show()

        # Learning Curve
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_scaled, y, cv=cv, scoring="accuracy",
            train_sizes=np.linspace(0.1, 1.0, 5)
        )

        plt.figure()
        plt.plot(train_sizes, train_scores.mean(axis=1), label="Train")
        plt.plot(train_sizes, test_scores.mean(axis=1), label="Validation")
        plt.title(f"SVM Learning Curve ({kernel}) ({dataset_name})")
        plt.xlabel("Training Size")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"images/svm_learning_curve_{kernel}_{dataset_name.lower()}.png", dpi=300)
        plt.show()

run_svm_b("data/bankruptcy_data.csv", target_column="Bankrupt?", dataset_name="Bankruptcy")
