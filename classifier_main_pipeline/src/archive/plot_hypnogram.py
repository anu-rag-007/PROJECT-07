import os
import matplotlib.pyplot as plt


def plot_hypnogram(stages, epoch_duration=30):

    stage_map = {
        "Wake": 0,
        "N1": 1,
        "N2": 2,
        "N3": 3,
        "REM": 4
    }

    y = [stage_map[s] for s in stages if s in stage_map]
    x = [i * epoch_duration for i in range(len(y))]

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.step(x, y, where="post")
    ax.set_yticks(list(stage_map.values()))
    ax.set_yticklabels(list(stage_map.keys()))
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Sleep Stage")
    ax.set_title("Hypnogram")

    os.makedirs("plots", exist_ok=True)
    save_path = os.path.join("plots", "hypnogram.png")

    print("Saving hypnogram to:", os.path.abspath(save_path))

    fig.canvas.draw()
    fig.savefig(save_path, dpi=300)

    plt.show()
    plt.close(fig)