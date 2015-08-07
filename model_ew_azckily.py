import fipir_aug
import sympy
# import numpy as np

reload(fipir_aug)

x_names = ['a', 'C', 'I', 'K', 'L', 'Y', 'zetaa']
x_dates = ['tm1', 't', 'tp1']
w_names = ['wa', 'wzetaa']
w_dates = ['t', 'tp1']
param_names = ['alpha', 'beta', 'delta', 'mua',
               'rhoa', 'nu', 'sigmaa', 'sigmazetaa']

param_names_module_to_user = {'rho_0': 'rhoa', 'mu_0': 'mua',
                              'sigma_signal_0': 'sigmaa',
                              'sigma_state_0': 'sigmazetaa'}               

xwp_sym_d = fipir_aug.make_x_w_param_sym_dicts(x_names, w_names, param_names)

x_s_d, w_s_d, param_sym_d = xwp_sym_d

x_in_ss_sym_d = {st: x_s_d[st] for st in x_s_d.keys() if 'ss' in st}

xss_ini_dict = {x_in_ss_sym_d['Y_ss']: 1.0,
                x_in_ss_sym_d['K_ss']: 2.0,
                x_in_ss_sym_d['zetaa_ss']: 0.01,
                x_in_ss_sym_d['a_ss']: 0.02,
                x_in_ss_sym_d['I_ss']: 0.3,
                x_in_ss_sym_d['C_ss']: 0.6,
                x_in_ss_sym_d['L_ss']: 0.5}

num_x_state_space = 1
num_s_state_space = 1

sspace_mat_sym = fipir_aug.make_state_space_sym(num_s_state_space,
                                                num_x_state_space, True)

##### parameter calibration part

alpha_value = 0.36
beta_value = 0.98
delta_value = 0.1
nu_value = 0.29
mua_value_true_Toy = 0.03
rhoa_value = 0.97  # 0.01087417, 0.00213658
sigmaza_value = 0.01*1
sigmaa_value = 0.01*7.5  # sskg = sskg= 0.128156077839
mua_value_SSFI = mua_value_true_Toy
mua_tm = mua_value_true_Toy

pref_tech_names_to_values = {'alpha': alpha_value, 'beta': beta_value,
                             'delta': delta_value, 'nu': nu_value}

ssp_par_user_to_values = {'mua': mua_value_true_Toy,
                                          'rhoa': rhoa_value,
                                          'sigmaa': sigmaa_value,
                                          'sigmazetaa': sigmaza_value}

ssp = fipir_aug.SignalStateSpace(n_x=num_x_state_space,
                                 n_s=num_s_state_space,
                                 param_val_dic=ssp_par_user_to_values,
                                 user_names_dic=param_names_module_to_user)

#ssp_par_module_to_values = {'mu_0': mua_value_true_Toy,
#                                          'rho_0': rhoa_value,
#                                          'sigma_signal_0': sigmaa_value,
#                                          'sigma_state_0': sigmaza_value}
#
#ssp_modnames = fipir_aug.SignalStateSpace(n_x=num_x_state_space,
#                                 n_s=num_s_state_space,
#                                 param_val_dic=ssp_par_module_to_values)

# ==============================================================================
#
# Define full information equilibrum conditions
# ==============================================================================


for k, v in x_s_d.iteritems():
    exec(k + '= v')

for k, v in w_s_d.iteritems():
    exec(k + '= v')

for k, v in param_sym_d.iteritems():
    exec(k + '= v')


def eq_conditions_toy_FI_CIKLY_az():

    It_expr = sympy.exp(at) * Ktp1 - (1-delta)*Kt
    Yt_expr = sympy.exp((1-alpha)*at) * Kt**(alpha) * Lt**(1-alpha)
    Ytp1_expr = sympy.exp((1-alpha)*atp1) * Ktp1**(alpha) * Ltp1**(1-alpha)

    at_expr = mua + zetaat + sigmaa*wat
    atp1_expr = mua + zetaatp1 + sigmaa*watp1

    zetaat_expr = rhoa*zetaatm1 + sigmazetaa*wzetaat

    g1 = beta*sympy.exp(-at) * (Ct/Ctp1) * \
        (alpha * Ytp1/Ktp1 + (1-delta)) - 1

    g2 = Yt - It - Ct

    g3 = nu*(1-alpha)*Yt/(Ct*(1-nu)) - Lt/(1-Lt)

    g4 = zetaat - zetaat_expr

    g5 = at - at_expr

    g6 = Yt - Yt_expr

    g7 = It - It_expr

    glist = [g1, g2, g3, g4, g5, g6, g7]

    return glist

glist_cikly_az = eq_conditions_toy_FI_CIKLY_az()
              



