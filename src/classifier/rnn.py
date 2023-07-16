import platform
from itertools import chain

import cv2
import numpy as np
import torch
from joblib import dump, load
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

system = platform.system()


class SimpleRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])  # Use the last output only
        return x


class SimpleLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers,
                 dropout, bidirectional):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size,
                            output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Use the last output only
        return x


def train(features):
    X = []
    y = []

    for _, v in features.items():
        for d in v:
            lbp_feature = list(chain(*d["lbp"]))

            feature = [
                d["aspect_ratio"],
                d["eccentricity"],
                d["extent"],
                d["solidity"],
                *d["hu_moments"],
                *lbp_feature,
            ]

            X.append(feature)
            y.append(d["class"])

    # Convert the lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Scale your data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Reshape X to fit RNN input shape
    X = X.reshape(-1, X.shape[1], 1)

    # Convert numpy arrays to PyTorch tensors
    X = torch.from_numpy(X).float()

    # Initialize the label encoder
    le = LabelEncoder()
    # Fit and transform the labels to encode them as integers
    y = le.fit_transform(y)
    # Convert labels to a torch tensor
    y = torch.from_numpy(y).long()

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=42,
                                                        stratify=y)

    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model = SimpleLSTM(
        input_size=X.shape[2],
        hidden_size=200,
        output_size=len(np.unique(y)),
        num_layers=2,
        dropout=0.5,
        bidirectional=True,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in range(100):  # Change number of epochs as needed
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

                # Evaluate on the test set for this epoch
                model.eval()
                with torch.no_grad():
                    test_preds = []
                    for inputs, _ in test_loader:
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs, 1)
                        test_preds.extend(predicted.numpy())
                test_acc = accuracy_score(y_test.numpy(), test_preds)
                print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # Save the model
    torch.save(model.state_dict(), "model.pt")


def classify(features, files):
    model = SimpleRNN(
        input_size=1, hidden_size=100,
        output_size=2)  # Ensure output_size matches your number of classes
    model.load_state_dict(torch.load("model.pt"))
    model.eval()  # Set the model to evaluation mode

    scaler = StandardScaler()  # Define the scaler

    for file in files:
        X = []

        # Your feature extraction logic here...
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

            X.append(feature)

        # Convert the lists to numpy arrays
        X = np.array(X)

        # Scale your data
        X = scaler.fit_transform(X)

        # Reshape X to fit RNN input shape
        X = X.reshape(-1, X.shape[1], 1)

        # Convert numpy arrays to PyTorch tensors
        X = torch.from_numpy(X).float()

        # Predict the classes
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        predictions = predicted.numpy()
        probabilities = nn.functional.softmax(outputs, dim=1).detach().numpy()

        print(predictions)
        # Continue with your remaining code...

        # Predict the classes
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
