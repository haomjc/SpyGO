from PyQt5 import QtWidgets
from pyvistaqt import QtInteractor
import pyvista as pv  # Import pyvista as pv

class MyApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyVistaQt Example")
        
        # Create a central widget and layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Add the PyVistaQt interactor
        self.plotter = QtInteractor(central_widget)
        layout.addWidget(self.plotter)
        
        # Add a mesh to the plotter
        self.plotter.add_mesh(pv.Sphere())

        # add a slider
        self.slider = QtWidgets.QSlider()
        self.slider.setOrientation(1)
        layout.addWidget(self.slider)
        self.slider.valueChanged.connect(self.on_slider_change)

    def on_slider_change(self, value):
        self.plotter.camera_position = [0, value/100, 0]
        print(value/100)
        # update view
        self.plotter.update()
        return
        
app = QtWidgets.QApplication([])
window = MyApp()
window.show()
app.exec_()