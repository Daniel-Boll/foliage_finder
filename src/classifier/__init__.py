import cv2
import numpy as np
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import platform

system = platform.system()

def train(features):
    # Separate the feature vectors and targets
    X = [{
        "aspect_ratio": d["aspect_ratio"],
        "eccentricity": d["eccentricity"],
        "extent": d["extent"],
        "hu_moments": d["hu_moments"],
        "solidity": d["solidity"],
    } for v in features.values() for d in v]

    # Ensure to flatten hu_moments and transform everything to a numerical format
    X = [[d["aspect_ratio"], d["eccentricity"], d["extent"], d["solidity"]] +
         list(d["hu_moments"]) for d in X]

    y = [d["class"] for _, v in features.items() for d in v]

    # Convert the lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Scale your data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    # Create the MLP classifier
    mlp = MLPClassifier(hidden_layer_sizes=(512, 256, 256), max_iter=8000)

    # Fit the model
    mlp.fit(X_train, y_train)

    # Save the model
    dump(mlp, "model.pkl")


def classify(features, files):
    mlp = load("model.pkl")

    for file in files:
        # Separate the feature vectors and targets
        X = [{
            "aspect_ratio": d["aspect_ratio"],
            "eccentricity": d["eccentricity"],
            "extent": d["extent"],
            "hu_moments": d["hu_moments"],
            "solidity": d["solidity"],
        } for d in features[file]]

        # Ensure to flatten hu_moments and transform everything
        # to a numerical format
        X = [
            [d["aspect_ratio"], d["eccentricity"], d["extent"], d["solidity"]
             ] + list(d["hu_moments"]) for d in X
        ]

        # Convert the lists to numpy arrays
        X = np.array(X)

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
        prob_of_class = np.max(probabilities, axis=1)

        for _, (i, leaf_features) in zip(features[file],
                                         enumerate(features[file])):
            print(
                f"Label: {predictions[i]} with probability {prob_of_class[i]}")

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
                f"{prob_of_class[i] * 100:.2f}%",
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