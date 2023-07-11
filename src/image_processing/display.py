import os

import cv2
import inquirer
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("tkagg")


def display_contours_from_features(features):
    questions = [
        inquirer.Text(
            "num_features",
            message="How many features to display/save? (leave blank for all)",
            default="",
        ),
        inquirer.List(
            "action",
            message="What action do you want to perform?",
            choices=["Show and Save", "Show Only", "Save Only"],
        ),
    ]

    answers = inquirer.prompt(questions)

    num_features = (None if answers["num_features"] == "" else int(
        answers["num_features"]))
    action = answers["action"]
    path = None

    if "Save" in action:
        path_question = [
            inquirer.Path(
                "path",
                message="Enter the path where you want to save the images",
                # Default cwd + /outputs/contours/
                default=os.path.join(os.getcwd(), "outputs", "contours"),
                exists=True,
                path_type=inquirer.Path.DIRECTORY,
            )
        ]
        path_answers = inquirer.prompt(path_question)
        path = path_answers["path"]

    # Counter to keep track of the number of processed features
    processed_features = 0

    for filename, feature in features.items():
        # Break the loop if we have processed the desired number of features
        if num_features is not None and processed_features >= num_features:
            break

        # Open the original image
        print(f"Processing {filename}")
        original_image = cv2.imread(filename)

        for feat in feature:
            contours = feat["contours"]

            for cnt in contours:
                cv2.drawContours(original_image, [cnt], 0, (128, 30, 128), 5)

            # Increment the counter
            processed_features += 1

        if "Save" in action:
            filename = filename.split("/")[-1]
            plt.imsave(
                f"{path}/contour_{filename}",
                cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
            )

        if "Show" in action:
            plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))


def plot_freeman_chain_code(chain_code, start=(0, 0)):
    # Define the moves for each direction
    moves = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0),
             (-1, 1)]

    # Initialize the position at the starting point
    pos = list(start)

    # Lists to hold the x and y coordinates of the points
    x_points = [start[0]]
    y_points = [start[1]]

    # For each direction in the chain code...
    for direction in chain_code:
        # Add the move for that direction to the current position
        move = moves[direction]
        pos[0] += move[0]
        pos[1] += move[1]

        # Add the new position to the lists
        x_points.append(pos[0])
        y_points.append(pos[1])

    # Plot the points
    plt.figure()
    plt.plot(x_points, y_points)
    plt.show()
