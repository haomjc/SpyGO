import tkinter as tk
from tkinter import ttk
import time

class Waitbar:
    def __init__(self, total_steps, title="Progress"):
        self.total_steps = total_steps
        self.current_step = 0
        
        # Create a Tkinter window
        self.window = tk.Tk()
        self.window.title(title)
        self.window.geometry("400x100")
        self.window.resizable(False, False)
        
        # Create a label
        self.label = tk.Label(self.window, text="Processing...", anchor="center")
        self.label.pack(pady=10)
        
        # Create a progress bar
        self.progress = ttk.Progressbar(self.window, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(pady=10)
        self.progress["maximum"] = total_steps
        
        # Update the GUI to show the window
        self.window.update()
    
    def update(self, step_increment=1, message="Processing..."):
        self.current_step += step_increment
        self.progress["value"] = self.current_step
        self.label.config(text=message)
        self.window.update()
    
    def close(self):
        self.window.destroy()

# Example usage
if __name__ == "__main__":
    total_steps = 100
    waitbar = Waitbar(total_steps, title="Processing Data")
    
    for i in range(total_steps):
        time.sleep(0.05)  # Simulate work being done
        waitbar.update(message=f"Processing step {i+1} of {total_steps}")
    
    waitbar.close()
