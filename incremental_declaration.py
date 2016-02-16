# incremental user program to test features of simplified BaseModel
import fipiro
import fipiro.utils as u
# import sympy
# from sympy.utilities.lambdify import lambdify
# from math import exp
# import numpy as np
reload(fipiro.utils)

u_x_names = ['K']
u_y_names = ['C', 'I', 'N', 'R', 'Y']
u_z_names = ['lnofZ']

x_names = u_x_names + u_y_names + u_z_names
w_names = ['epsilon']
param_names = ['A', 'beta', 'delta', 'psi',
               'rho', 'eta', 'sigma']

v_names = {'x_names': x_names, 'w_names': w_names}

bmodel = u.BaseModel(var_names_dict=v_names, param_names=param_names)
