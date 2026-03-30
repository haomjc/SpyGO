"""
Example usage of the ease-off functionality in the gears package

This script demonstrates how to use the EaseOffGenerator class to create
and visualize gear tooth micro-geometry modifications.
"""
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gears.main.ease_off import crowning_RZE


def main():
    """Demonstrate ease-off functionality"""
    print("=== Gears Package Ease-off Example ===\n")
    
    # Define typical gear parameters
    ra = 21#25.0           # Addendum radius (tip radius) [mm]
    rff = 20.54578      # Form diameter radius (fillet-to-involute transition) [mm]  
    rf = 18.75#18.0               # Root radius [mm]
    b = 20.0                # Face width [mm]
    
    # Define ease-off parameters
    Ep = 15.0    # Profile ease-off amplitude [μm]
    Er = 35.0
    Et = 8.0         # Toe ease-off amplitude [μm]
    Eh = 12.0       # Heel ease-off amplitude [μm]
    
    E = crowning_RZE(ra,rff,rf,b,Ep,Er,Et,Eh,3,3)

    R = np.linspace(rf,ra,50)
    z = np.linspace(-b/2, b/2, 20)

    zz,rr = np.meshgrid(z, R)

    E_val = E(zz,rr)

    # Keep the script alive to let user interact with plots
    import matplotlib.pyplot as plt

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(zz, rr, E_val, cmap='viridis')

    # Show the plot
    plt.show()

    try:
        input("\nPress Enter to exit (this keeps the plot windows open)...")
    except KeyboardInterrupt:
        print("\n   Exiting...")
    finally:
        plt.close('all')  # Clean up all plot windows

if __name__ == '__main__':
    main()