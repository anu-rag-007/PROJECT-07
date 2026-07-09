def classify_stage(features):
    ratio = features["ratio"]
    delta = features["delta"]

    if ratio > 1.2:
        return "Wake"
    elif ratio > 0.8:
        return "N1"
    elif delta < 0.5:
        return "N2"
    else:
        return "N3"
