# -*- coding: utf-8 -*-

import numpy as np
import data

td = data.load_kwh_data()
cov = np.cov(td)
print(td)
print(cov)

