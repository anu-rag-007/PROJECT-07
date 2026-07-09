import csv
import os
from datetime import datetime

def log_event(epoch, rem_prob, ratio, vibration):

    file_exists = os.path.isfile("logs/sleep_log.csv")

    with open("logs/sleep_log.csv", "a", newline="") as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "epoch",
                "rem_probability",
                "theta_delta_ratio",
                "vibration_triggered"
            ])

        writer.writerow([
            datetime.now(),
            epoch,
            rem_prob,
            ratio,
            vibration
        ])