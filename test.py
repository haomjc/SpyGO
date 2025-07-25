from hypoid.main.dataclasses import DesignData, FlankNumericalData
from general_utils import dataclass_print, dataclass_to_file

import numpy as np
import copy
designData = DesignData()
# dataclass_print(designData)

print(designData.gear_common_data.BACKANGLE)

flankData = FlankNumericalData()

print(flankData.gear.concave)

