import sounddevice as sd
import json

# Query input devices and print in JSON format
def print_input_devices_json():
    devices = sd.query_devices(kind='input')
    print(json.dumps(devices, indent=4))

# Main function to execute the script
if __name__ == "__main__":
    print_input_devices_json()