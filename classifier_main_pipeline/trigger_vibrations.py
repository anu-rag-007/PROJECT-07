import requests
import time

JOIN_API_KEY = "d0351db08bc94ec89fd9cebbb0a7a785"
DEVICE_ID = "9ca52c361cae4532bca975309e6856ce"

def send(command):
    url = "https://joinjoaomgcd.appspot.com/_ah/api/messaging/v1/sendPush"
    
    data = {
        "apikey": JOIN_API_KEY,
        "deviceId": DEVICE_ID,
        "text": command,
        "title": command
    }

    try:
        response = requests.post(url, data=data)
        print(f"Sent: {command} | Status:", response.status_code)
        print("Response:", response.text)
    except Exception as e:
        print("Error:", e)

print("Starting test...")

send("walk_start")
time.sleep(5)
send("walk_stop")

print("Test completed")