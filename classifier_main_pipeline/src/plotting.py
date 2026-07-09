import matplotlib.pyplot as plt

def plot_hypnogram(stages, save_path=None):

    stage_mapping = {
        "Wake": 4,
        "REM": 3,
        "N1": 2,
        "N2": 1,
        "N3": 0
    }

    numeric_stages = [stage_mapping[s] for s in stages]

    plt.figure(figsize=(10, 4))
    plt.step(range(len(numeric_stages)), numeric_stages, where="post")
    plt.yticks(
        [4, 3, 2, 1, 0],
        ["Wake", "REM", "N1", "N2", "N3"]
    )
    plt.xlabel("Epoch")
    plt.ylabel("Sleep Stage")
    plt.title("Predicted Hypnogram")

    plt.gca().invert_yaxis()

    if save_path is not None:
        plt.savefig(save_path)
        print(f"Hypnogram saved at: {save_path}")

    plt.show()