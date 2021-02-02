# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 09:15:37 2021

@author: eugen
"""

import numpy as np

# The python list
py_arr = [1,2,3,4,5]

# The NumPy Array of size 5
np_arr = np.zeros(5)

np_arr = np.zeros((5,2))

# Put 1 to the 5 * 2 Matrix
np.put(np_arr, range(6), np.ones((3,2)))

#Or copy contents from single or multi Dimensional Arrays
#using one of the following ways

np.put(np_arr, range(10), [1,2,3,4,5,6,7,8,9,0])
np.put(np_arr, range(10), [[1,2,3,4,5],[6,7,8,9,0]])
np.put(np_arr, range(10), [[1,2], [3,4], [5,6], [7,8], [9,10]])
