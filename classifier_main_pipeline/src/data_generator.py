import numpy as np


def generate_stage_signal(stage, length=100):
    t = np.linspace(0, 1, length)

    if stage == "Wake":
        signal = 0.5 * np.sin(2 * np.pi * 20 * t)
        signal += np.random.normal(0, 0.5, length)

    elif stage == "N1":
        signal = 0.8 * np.sin(2 * np.pi * 6 * t)
        signal += np.random.normal(0, 0.3, length)

    elif stage == "N2":
        signal = 0.8 * np.sin(2 * np.pi * 6 * t)
        if np.random.rand() > 0.5:
            signal += 0.6 * np.sin(2 * np.pi * 13 * t)
        signal += np.random.normal(0, 0.3, length)

    elif stage == "N3":
        signal = 2.0 * np.sin(2 * np.pi * 2 * t)
        signal += np.random.normal(0, 0.2, length)

    elif stage == "REM":
        signal = 0.6 * np.sin(2 * np.pi * 10 * t)
        signal += np.random.normal(0, 0.4, length)

    return signal


def generate_sleep_cycle():

    cycle = ["Wake"]

    # Light sleep entry
    cycle += ["N1"]
    cycle += ["N2"] * np.random.randint(1, 3)

    # Deep sleep may or may not happen
    if np.random.rand() > 0.3:
        cycle += ["N3"] * np.random.randint(1, 3)

    cycle += ["N2"]

    # REM duration varies
    cycle += ["REM"] * np.random.randint(1, 3)

    # Random short wake intrusion
    if np.random.rand() > 0.6:
        cycle += ["Wake"]

    return cycle



def generate_dataset(num_cycles=200, length=100):

    X = []
    y = []

    for _ in range(num_cycles):

        cycle = generate_sleep_cycle()

        for stage in cycle:
            signal = generate_stage_signal(stage, length)

            X.append(signal)
            y.append(stage)

    return np.array(X), np.array(y)
