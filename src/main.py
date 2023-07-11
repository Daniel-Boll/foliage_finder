import sys

import classifier
import image_processing
from utils.files import get_files_from_dir


def train():
    files = get_files_from_dir("labeled/", "png")
    features = image_processing.extract_features(files,
                                                 image_processing.Step.TRAIN)
    classifier.train(features)


def classify():
    files = get_files_from_dir("custom/", "bmp")
    features = image_processing.extract_features(
        files, image_processing.Step.CLASSIFY)
    # image_processing.display(features)
    classifier.classify(features, files)


def main():
    try:
        # train()
        classify()

    except FileNotFoundError as e:
        print(e, file=sys.stderr)


if __name__ == "__main__":
    main()
