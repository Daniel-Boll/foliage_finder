import pprint
import sys

import classifier
import image_processing
from utils.files import get_files_from_dir

pp = pprint.PrettyPrinter(indent=2, depth=1, width=40, compact=True)


def train():
    files = get_files_from_dir("labeled/", "png")
    features = image_processing.extract_features(
        # image_paths=files,
        image_paths=files[:1],
        step=image_processing.Step.TRAIN,
        augmentation_times=1)
    pp.pprint(features)
    # classifier.train(features)


def classify():
    files = get_files_from_dir("custom/", "bmp")
    features = image_processing.extract_features(
        files, image_processing.Step.CLASSIFY)
    # image_processing.display(features)
    classifier.classify(features, files)


def main():
    try:
        train()
        # classify()

    except FileNotFoundError as e:
        print(e, file=sys.stderr)


if __name__ == "__main__":
    main()
