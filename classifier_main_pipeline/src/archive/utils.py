import csv
import os

def save_sleep_log(filename, rows):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_sec",
            "raw_stage",
            "smooth_stage",
            "final_stage",
            "confidence"
        ])
        writer.writerows(rows)
