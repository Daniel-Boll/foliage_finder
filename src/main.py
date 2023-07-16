import pprint
import sys

import classifier
import image_processing
from utils.files import get_balanced_files, get_files_from_dir

pp = pprint.PrettyPrinter(indent=2, depth=1, width=40, compact=True)


def preprocess():
    files = get_files_from_dir("labeled/uva_do_mato", "png")
    image_processing.augment_images(
        image_paths=files,
        augmentation_times=40,
    )


def train():
    files = get_files_from_dir("labeled/", "png")
    # files = get_balanced_files("labeled/", "png", 40)
    features = image_processing.extract_features(
        image_paths=files,
        step=image_processing.Step.TRAIN,
    )
    # Save the features to a file
    image_processing.save_features(features)
    # features = image_processing.load_features()
    classifier.mlp.train(features)


def classify():
    files = get_files_from_dir("custom/", "bmp")
    features = image_processing.extract_features(
        image_paths=files, step=image_processing.Step.CLASSIFY)
    # print(features)
    # image_processing.display(features)
    # classifier.rnn.classify(features, files)
    # classifier.mlp.classify(features, files)
    classifier.mlp.classify_per_feature(features, files)


def main():
    try:
        # preprocess()
        # train()
        classify()

    except FileNotFoundError as e:
        print(e, file=sys.stderr)


if __name__ == "__main__":
    main()
