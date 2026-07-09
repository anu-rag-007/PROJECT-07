import csv
import os

def save_sleep_log(filepath, rows):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Time(sec)",
            "Raw Stage",
            "Smoothed Stage",
            "Final Stage",
            "Confidence"
        ])
        writer.writerows(rows)
