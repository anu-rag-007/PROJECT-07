import numpy as np
import joblib
import requests
import time
from collections import deque
import os
import csv
import datetime

# ==============================
# CONFIG
# ==============================

JOIN_API_KEY = "d0351db08bc94ec89fd9cebbb0a7a785"
DEVICE_ID = "9ca52c361cae4532bca975309e6856ce"

WINDOW_SIZE = 3            # stability (3 epochs)
REM_DELAY = 60             # seconds before stimulation
BURST_DURATION = 15        # seconds vibration ON
PAUSE_DURATION = 45        # seconds OFF
MAX_CYCLES = 3             # repeat bursts

# ==============================
# LOAD MODEL
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "rem_model.pkl")

model = joblib.load(MODEL_PATH)

# ==============================
# STATE
# ==============================

rem_buffer = deque(maxlen=WINDOW_SIZE)
in_rem = False
stimulation_running = False

# ==============================
# JOIN FUNCTION
# ==============================

def send(cmd):
    url = "https://joinjoaomgcd.appspot.com/_ah/api/messaging/v1/sendPush"
    
    data = {
        "apikey": JOIN_API_KEY,
        "deviceId": DEVICE_ID,
        "text": cmd
    }

    try:
        r = requests.post(url, data=data)
        print(f"[JOIN] {cmd} | {r.status_code}")
    except Exception as e:
        print("Join Error:", e)

# ==============================
# STIMULATION LOGIC
# ==============================

def run_stimulation():
    global stimulation_running

    print("Starting burst stimulation")
    stimulation_running = True
    
    log_event("STIM_START")

    for i in range(MAX_CYCLES):

        print(f"Cycle {i+1}")

        send("walk_start")
        time.sleep(BURST_DURATION)

        send("walk_stop")
        time.sleep(PAUSE_DURATION)

    print("Stimulation complete")
    stimulation_running = False
    
    log_event("STIM_STOP")

# ==============================
# MAIN PROCESS FUNCTION
# ==============================

def process_epoch(features):
    global in_rem, stimulation_running

    pred = model.predict(features.reshape(1, -1))[0]

    print("Prediction:", pred)

    rem_buffer.append(pred)
    print("Buffer:", list(rem_buffer))

    # -----------------------------
    # Stable REM Detection
    # -----------------------------
    if len(rem_buffer) == WINDOW_SIZE and all(rem_buffer):

        if not in_rem:
            print("Stable REM detected")
            in_rem = True
            
            log_event("REM_DETECTED")

            print(f"Waiting {REM_DELAY}s...")
            time.sleep(REM_DELAY)

            if not stimulation_running:
                run_stimulation()

    # -----------------------------
    # Exit REM
    # -----------------------------
    if pred == 0:
        if in_rem:
            print("REM ended")
            
            log_event("REM_END")

        in_rem = False
        rem_buffer.clear()

        if stimulation_running:
            print("Force stop")
            send("walk_stop")
            stimulation_running = False

# ==============================
# TEST MODE
# ==============================

if __name__ == "__main__":

    print("=== TEST MODE ===")

    # Simulated predictions
    test_seq = [0,0,1,1,1,1,1,1,0]

    dummy = np.random.rand(10)

    for p in test_seq:

        rem_buffer.append(p)
        print("Simulated Buffer:", list(rem_buffer))

        if len(rem_buffer) == WINDOW_SIZE and all(rem_buffer):
            print("TEST REM")

            time.sleep(5)
            run_stimulation()

            rem_buffer.clear()

        time.sleep(1)
        
def log_event(event, value=""):
    with open("experiment_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.datetime.now(),
            event,
            value
        ])