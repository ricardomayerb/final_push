import fipiro
import fipiro.utils as u
import sympy
# from sympy.utilities.lambdify import lambdify
# from math import exp
import numpy as np
reload(fipiro.utils)


x_names = ['C', 'I', 'K', 'N', 'Y', 'Z']
x_dates = ['_tm1', '_t', '_tp1']
w_names = ['wa', 'wzetaa']
w_dates = ['_t', '_tp1']
param_names = ['alpha', 'beta', 'delta', 'mua',
               'rhoa', 'nu', 'sigmaa', 'sigmazetaa']
all_names = {
    'x_names': x_names, 'w_names': w_names, 'param_names': param_names}

xxsswp_sym_d = u.make_x_w_param_sym_dicts(x_names, w_names, param_names)
x_s_d, x_in_ss_sym_d, w_s_d, param_sym_d = xxsswp_sym_d
xss_ini_dict = {x_in_ss_sym_d['Y_ss']: 1.0,
                x_in_ss_sym_d['K_ss']: 2.0,
                x_in_ss_sym_d['Z_ss']: 0.01,
                x_in_ss_sym_d['I_ss']: 0.3,
                x_in_ss_sym_d['C_ss']: 0.6,
                x_in_ss_sym_d['N_ss']: 0.5}

xxsswp_sym_d = u.make_x_w_param_sym_dicts(x_names, w_names, param_names)
x_s_d, x_in_ss_sym_d, w_s_d, param_sym_d = xxsswp_sym_d

def g_hansen_1985_div_tk():
    eq1 = C_t - 1

    return [eq1]

def g_hansen_1985_div_miao():
    eq1 = It -2

    return [eq1]

    
