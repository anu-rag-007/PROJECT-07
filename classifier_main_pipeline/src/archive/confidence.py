def compute_confidence(features):
    """
    Confidence reflects how clearly the EEG matches a sleep stage.
    Higher delta/theta dominance → higher confidence.
    """
    delta = features["delta"]
    theta = features["theta"]
    alpha = features["alpha"]
    beta  = features["beta"]

    numerator = delta + theta
    denominator = alpha + beta + 1e-6  # avoid divide by zero

    confidence = numerator / denominator
    return min(confidence, 1.0)
