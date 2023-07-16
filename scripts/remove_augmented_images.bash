#!/bin/bash
# This script will delete all files with names containing "_aug_" in the "labeled" directory and its subdirectories.

find ./labeled/ -type f -name '*_aug_*' -exec rm {} \;
