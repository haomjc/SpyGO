from hypoid.main.dataclasses import initialize_design_data, numerical_template_data
from general_utils import dataclass_print, dataclass_to_file

import numpy as np
import copy
designData = initialize_design_data()
# dataclass_print(designData)

templateData = numerical_template_data()
templateData2 = copy.deepcopy(templateData)
templateData.gear.convex = np.array([1.0, 2.0, 3.0])
dataclass_print(templateData)

st = 'hello'
print(st + ' ' * 20 + st)
file = 'test_output.txt'
dataclass_to_file(designData, file, indent=3, write=True)

print(getattr(getattr(designData, 'gear_common_data'), 'OUTERCONEDIST'))