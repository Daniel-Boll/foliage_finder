import numpy as np


def extract_features(contours):
    max_len = 0
    chain_codes = []
    for contour in contours:
        chain_code = calculate_freeman_chain_code(
            list(map(tuple, np.squeeze(contour, axis=1))))
        chain_codes.append(chain_code)
        max_len = max(max_len, len(chain_code))

    # Pad all chain codes to have the same length
    padded_chain_codes = [cc + [0] * (max_len - len(cc)) for cc in chain_codes]

    return padded_chain_codes


def calculate_freeman_chain_code(contour):
    # Directions for Freeman chain code
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0),
                  (-1, 1)]

    # Subtract each point by the first point
    contour = [(p[0] - contour[0][0], p[1] - contour[0][1]) for p in contour]

    # Divide each point by the maximum absolute value in the contour
    max_value = max(max(abs(x), abs(y)) for x, y in contour)
    contour = [(x / max_value, y / max_value) for x, y in contour
               if max_value != 0 and not np.isnan(max_value)]

    # Round each point to the nearest direction and convert to Freeman chain code
    freeman_code = []
    for x, y in contour:
        min_distance = float("inf")
        direction = None
        for d, (dx, dy) in enumerate(directions):
            distance = (x - dx)**2 + (y - dy)**2
            if distance < min_distance:
                min_distance = distance
                direction = d
        if not freeman_code or freeman_code[-1] != direction:
            freeman_code.append(direction)

    return freeman_code
