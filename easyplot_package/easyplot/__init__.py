# easyplot/__init__.py

from .plot2d import line, scatter
from .plot3d import surface, patch
from .transforms import HgTransform2D, HgTransform3D

__all__ = ['line', 'scatter', 'surface', 'patch', 'HgTransform2D', 'HgTransform3D']
