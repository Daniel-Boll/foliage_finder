import platform
from itertools import chain
from typing import Dict, List

import cv2
import numpy as np
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

system = platform.system()


def train(features):
    X = []
    y = []

    for _, v in features.items():
        for d in v:
            for i in range(len(d["extent"])):
                feature = [
                    d["aspect_ratio"][i],
                    d["eccentricity"][i],
                    d["extent"][i],
                    d["solidity"][i],
                    *d["hu_moments"][i],
                    *d["lbp"][i],
                ]

                X.append(feature)
                y.append(d["class"])

    # Convert the lists to numpy arrays
    X = np.array(X)
    print(X.shape)
    y = np.array(y)

    # Scale your data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    # Create the MLP classifier
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100),
                        max_iter=8000,
                        verbose=True)

    # Fit the model
    mlp.fit(X_train, y_train)

    # Save the model
    dump(mlp, "model.pkl")


def classify(features, files):
    mlp = load("model.pkl")

    count = 0
    for file in files:
        X = []

        for d in features[file]:
            lbp_feature = list(chain(*d["lbp"]))

            feature = [
                d["aspect_ratio"],
                d["eccentricity"],
                d["extent"],
                d["solidity"],
                *d["hu_moments"],
                *lbp_feature,
            ]

            if count == 0:
                print(feature)
                count += 1
            X.append(feature)

        # Convert the lists to numpy arrays
        X = np.array(X)
        print(X.shape)

        # Scale your data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Predict the classes
        predictions = mlp.predict(X)
        probabilities = mlp.predict_proba(X)

        print(predictions)
        print(probabilities)

        image = cv2.imread(file)

        # Get the scalar greates probability for each class
        probability_of_class = np.max(probabilities, axis=1)

        for i, leaf_features in enumerate(features[file]):
            print(
                f"Label: {predictions[i]} with probability {probability_of_class[i]}"
            )

            print("Features:")
            leaf_contour = leaf_features["contours"][i]

            # Get the bounding rectangle
            x, y, w, h = cv2.boundingRect(leaf_contour)

            # Calculate the center of the rectangle
            class_label = (x + w // 2, y + h // 2)
            percentage_label = (x + w // 2, y + h // 2 + 40)

            # Draw the text label on the image
            cv2.putText(
                image,
                str(predictions[i]),
                class_label,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Draw the percentage label on the image
            cv2.putText(
                image,
                f"{probability_of_class[i] * 100:.2f}%",
                percentage_label,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )

        # Display the image
        # cv2.imshow("Image with Labels", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if system == "Windows":
            cv2.imwrite("classified/" + file.split("\\")[-1], image)
        else:
            cv2.imwrite("classified/" + file.split("/")[-1], image)


def classify_per_feature(images_features: Dict[str, List[Dict]],
                         files: List[str]) -> None:
    """
    Loads a model and use it to classify images based on multiple features for each leaf in the image.

    :param images_features: a dictionary where the key is the file name and
                            the value is a list of dictionaries. Each dictionary
                            contains keys corresponding to different features
                            and values are lists of these features for each leaf in the image.
    :param files: a list of file names to process.
    :return: None
    """
    mlp = load("model.pkl")
    # Iterate over the files, the for each file get its set of features.
    # The set of features is a dictorionary where each key is a feature
    # and the value is a list of features for each leaf in the image

    for file in files:
        X = []
        contours = []
        for i, features in enumerate(images_features[file]):
            for j in range(len(features["lbp"])):
                feature = [
                    features["aspect_ratio"][j],
                    features["eccentricity"][j],
                    features["extent"][j],
                    features["solidity"][j],
                    *features["hu_moments"][j],
                    *features["lbp"][j],
                ]

                X.append(feature)
                contours.append(features["contours"][j])

        X = np.array(X)

        # Scale your data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Predict the classes
        predictions = mlp.predict(X)
        probabilities = mlp.predict_proba(X)

        image = cv2.imread(file)
        for i, (prediction,
                probability) in enumerate(zip(predictions, probabilities)):
            probability_of_class = np.max(probability)
            print(
                f"Label: {prediction} with probability {probability_of_class}")

            leaf_contour = contours[i]

            # Get the bounding rectangle
            x, y, w, h = cv2.boundingRect(leaf_contour)

            # Calculate the center of the rectangle
            class_label = (x + w // 2, y + h // 2)
            percentage_label = (x + w // 2, y + h // 2 + 40)

            # Draw the text label on the image
            cv2.putText(
                image,
                str(prediction),
                class_label,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Draw the percentage label on the image
            cv2.putText(
                image,
                f"{probability_of_class * 100:.2f}%",
                percentage_label,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )

        cv2.imwrite("classified/" + file.split("/")[-1], image)
