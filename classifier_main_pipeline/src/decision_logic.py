# decision_logic.py

from typing import List


def run_closed_loop(
    stages: List[str],
    epoch_duration: int = 30,
    min_n1_duration: int = 30,
    no_wake_window: int = 60,
) -> List[int]:
    """
    Closed-loop decision logic for N1-triggered stimulation.

    Parameters:
        stages (List[str]): Sleep stage labels in chronological order.
        epoch_duration (int): Seconds per epoch (default 30).
        min_n1_duration (int): Required stable N1 duration in seconds.
        no_wake_window (int): Time window before trigger where no 'Wake' must appear.

    Returns:
        List[int]: List of trigger timestamps in seconds.
    """

    trigger_times = []

    min_n1_epochs = min_n1_duration // epoch_duration
    no_wake_epochs = no_wake_window // epoch_duration

    n1_counter = 0
    stimulation_used = False

    for i in range(len(stages)):
        current_stage = stages[i]

        # Track continuous N1 duration
        if current_stage == "N1":
            n1_counter += 1
        else:
            n1_counter = 0

        # Check trigger conditions
        if (
            n1_counter >= min_n1_epochs
            and not stimulation_used
        ):
            recent_window = stages[max(0, i - no_wake_epochs):i]

            if "Wake" not in recent_window:
                trigger_time = i * epoch_duration
                trigger_times.append(trigger_time)

                stimulation_used = True
                n1_counter = 0  # prevent immediate re-trigger

    return trigger_times