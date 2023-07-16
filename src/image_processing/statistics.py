from typing import List

import cv2
import numpy as np


def extract_features(contours: List[np.ndarray]):
    contour_features = []
    aspect_ratios = []
    eccentricities = []
    solidities = []
    extents = []
    hu_moments_ = []

    for c in contours:
        if len(c) < 5:  # Skip contours with fewer than 5 points
            continue
        # Calculate the Aspect Ratio
        _x, _y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h

        # Calculate Solidity and Extent
        area = cv2.contourArea(c)
        _x, _y, w, h = cv2.boundingRect(c)
        rect_area = w * h
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area
        extent = float(area) / rect_area

        # Calculate Eccentricity
        ((_x, _y), (MA, ma), _angle) = cv2.fitEllipse(c)
        eccentricity = np.sqrt(1 - (MA / ma)**2)

        # Calculate Hu Moments
        moments = cv2.moments(c)
        hu_moments = cv2.HuMoments(moments)
        # Log transform to make hu moments scale invariant
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))

        aspect_ratios.append(aspect_ratio)
        eccentricities.append(eccentricity)
        solidities.append(solidity)
        extents.append(extent)
        hu_moments_.append(hu_moments.flatten())

    # contour_features.append({
    #     "aspect_ratio": aspect_ratio,
    #     "eccentricity": eccentricity,
    #     "solidity": solidity,
    #     "extent": extent,
    #     "hu_moments": hu_moments.flatten(),
    #     "contours": contours,
    # })
    contour_features.append({
        "aspect_ratio": aspect_ratios,
        "eccentricity": eccentricities,
        "solidity": solidities,
        "extent": extents,
        "hu_moments": hu_moments_,
        "contours": contours,
    })

    return contour_features
