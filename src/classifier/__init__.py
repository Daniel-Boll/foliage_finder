import cv2
import numpy as np
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import platform

system = platform.system()

def train(features):
    X = []
    y = []

    for _, v in features.items():
        for d in v:
            feature = [
                d["aspect_ratio"],
                d["eccentricity"],
                d["extent"],
                d["solidity"],
                *d["hu_moments"],
            ]

            X.append(feature)
            y.append(d["class"])

    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlp = MLPClassifier(hidden_layer_sizes=(512, 256, 256), max_iter=8000)
    mlp.fit(X_train, y_train)

    dump(mlp, "model.pkl")


def classify(features, files):
    mlp = load("model.pkl")

    for file in files:
        X = []

        for d in features[file]:
            feature = [
                d["aspect_ratio"],
                d["eccentricity"],
                d["extent"],
                d["solidity"],
                *d["hu_moments"],
            ]

            X.append(feature)

        X = np.array(X)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        predictions = mlp.predict(X)
        probabilities = mlp.predict_proba(X)

        image = cv2.imread(file)

        prob_of_class = np.max(probabilities, axis=1)

        for i, leaf_features in enumerate(features[file]):
            print(f"Label: {predictions[i]} with probability {prob_of_class[i]}")
            print("Features:")
            leaf_contour = leaf_features["contours"][i]
            x, y, w, h = cv2.boundingRect(leaf_contour)
            class_label = (x + w // 2, y + h // 2)
            percentage_label = (x + w // 2, y + h // 2 + 40)

            cv2.putText(
                image,
                str(predictions[i]),
                class_label,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                image,
                f"{prob_of_class[i] * 100:.2f}%",
                percentage_label,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )

        if system == "Windows":
            cv2.imwrite("classified/" + file.split("\\")[-1], image)
        else:
            cv2.imwrite("classified/" + file.split("/")[-1], image)
