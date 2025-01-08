import time
import os

def run_server():
    print("Server started. Waiting for commands...")
    while True:
        # Simulate waiting for commands
        command = input("Enter command (or 'exit' to stop): ")
        if command == "exit":
            print("Shutting down server.")
            break
        print(f"Executing command: {command}")
        eval(command)
        time.sleep(1)

if __name__ == "__main__":
    run_server()