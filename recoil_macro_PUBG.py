import time
import mouse
import threading
import keyboard
import ctypes
import queue

# Define necessary constants
MOUSEEVENTF_MOVE = 0x0001

# Flag to control the script execution
running = False  
enabled = False  # This controls whether the mouse event is active
move_y = 10
lock = threading.Lock()

# Queue for communication between threads
event_queue = queue.Queue()

def run_script_continuous():
    global running
    if running:
        return  # Prevent multiple instances from starting
    running = True
    
    try:
        while mouse.is_pressed("left"):  # Keep running while button is held
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE, 0, move_y, 0, 0)
            time.sleep(0.05)  # Adjust the delay as needed
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Mouse button released. Stopping script.")
        running = False  # Reset flag when button is released

# Toggle function to enable/disable mouse listening
def toggle_mouse_event():
    global enabled
    with lock:
        enabled = not enabled  # Toggle the flag
        status = "ENABLED" if enabled else "DISABLED"
        print(f"Mouse event {status}")
        if not enabled:
            mouse.unhook_all()  # Stop mouse listening when disabled

def increase_y():
    global move_y
    with lock:
        move_y += 1
        print(f"Move y: {move_y}") 

def decrease_y():
    global move_y
    with lock:
        print(f"Move y: {move_y}")

# Listen for keybinding to toggle the mouse event
keyboard.add_hotkey("\\", toggle_mouse_event)  # Press "\\" to toggle
keyboard.add_hotkey("+", increase_y)  # Press "+" to increase move_y
keyboard.add_hotkey("-", decrease_y)  # Press "-" to decrease move_y

# Mouse event listener function
def on_mouse_button(event):
    # check if event is ButtonEvent, otherwise ignore
    if not isinstance(event, mouse.ButtonEvent):
        return

    if enabled and event.button == 'left' and event.event_type == 'down':
        # threading.Thread(target=run_script_continuous).start()
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE, 0, move_y, 0, 0)
        time.sleep(0.05)  # Adjust the delay as needed

# Main loop: Activate mouse listener only if enabled
def mouse_listener():
    while True:
        if enabled:
            mouse.hook(on_mouse_button)  # Hook into mouse events
            # event_queue.put("Listening")  # For debug purposes
    # event_queue.put("Idle")  # Debugging: show status of listener
        time.sleep(0.1)  # Avoid high CPU usage in the main loop

# Start the mouse listener in a background thread
listener_thread = threading.Thread(target=mouse_listener, daemon=True)
listener_thread.start()

print("Press '\\' to toggle mouse functionality. Press Ctrl+C to exit.")

# Keep the script running
try:
    while True:
        if not event_queue.empty():
            status = event_queue.get()
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nStopped listening.")