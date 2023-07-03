import pprint
import sys

import image_processing
from utils.files import get_files_from_dir


def main():
    pp = pprint.PrettyPrinter(indent=2)

    try:
        files = get_files_from_dir("assets/", "bmp")
        # features = image_processing.extract_features(files[:2])
        features = image_processing.extract_features(files)
        image_processing.display(features)


    except FileNotFoundError as e:
        print(e, file=sys.stderr)


if __name__ == "__main__":
    main()
