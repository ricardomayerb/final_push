#import fipir_aug
import fipiro
import sympy
import numpy as np
from scipy import linalg

#reload(fipir_aug)
reload(fipiro)

x_names = ['a', 'C', 'I', 'K', 'L', 'Y', 'zetaa']
x_dates = ['tm1', 't', 'tp1']
w_names = ['wa', 'wzetaa']
w_dates = ['t', 'tp1']
param_names = ['alpha', 'beta', 'delta', 'mua',
               'rhoa', 'nu', 'sigmaa', 'sigmazetaa']
all_names = {'x_names':x_names,'w_names':w_names,'param_names':param_names}

param_names_module_to_user = {'rho_0': 'rhoa', 'mu_0': 'mua',
                              'sigma_signal_0': 'sigmaa',
                              'sigma_state_0': 'sigmazetaa'}               

xxsswp_sym_d = fipiro.make_x_w_param_sym_dicts(x_names, w_names, param_names)

x_s_d, x_in_ss_sym_d, w_s_d, param_sym_d = xxsswp_sym_d

xss_ini_dict = {x_in_ss_sym_d['Y_ss']: 1.0,
                x_in_ss_sym_d['K_ss']: 2.0,
                x_in_ss_sym_d['zetaa_ss']: 0.01,
                x_in_ss_sym_d['a_ss']: 0.02,
                x_in_ss_sym_d['I_ss']: 0.3,
                x_in_ss_sym_d['C_ss']: 0.6,
                x_in_ss_sym_d['L_ss']: 0.5}

num_x_state_space = 1
num_s_state_space = 1

sspace_mat_sym = fipiro.make_state_space_sym(num_s_state_space,
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

ssp = fipiro.SignalStateSpace(n_x=num_x_state_space,
                                 n_s=num_s_state_space,
                                 param_val_dic=ssp_par_user_to_values,
                                 user_names_dic=param_names_module_to_user)

#ssp_par_module_to_values = {'mu_0': mua_value_true_Toy,
#                                          'rho_0': rhoa_value,
#                                          'sigma_signal_0': sigmaa_value,
#                                          'sigma_state_0': sigmaza_value}
#
#ssp_modnames = fipiro.SignalStateSpace(n_x=num_x_state_space,
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

utility_elw = nu * sympy.log(Ct) + (1-nu) * sympy.log(Lt)

all_param_values_dict = {}
all_param_values_dict.update(pref_tech_names_to_values)
all_param_values_dict.update(ssp_par_user_to_values)

fi = fipiro.FullInfoModel(ssp, all_names,
                             par_to_values_dict=all_param_values_dict,
                             eq_conditions=glist_cikly_az,
                             utility=utility_elw,
                             xss_ini_dict=xss_ini_dict)
              
dfi_unev, dse_unev = fi.d1d2_g_x_w_unevaluated()

fun1n, fun2n, vn = fi.make_numpy_fns_of_d1d2xw(dfi_unev, dse_unev)


#fun1t, fun2t, vt = fi.make_theano_fns_of_d1d2xw(dfi_unev, dse_unev)

dfi_n, dse_n = fi.get_evaluated_dgdxw12(mod='numpy')

##dfi_t, dse_t = fi.get_evaluated_dgdxw12(mod='theano')
#
psi_x, psi_w, psi_q =  fi.get_first_order_approx_coeff_fi()

psi_second_order = fi.get_second_order_approx_coeff_fi(psi_x, psi_w, psi_q, dfi_n, dse_n)

psi_x_x, psi_x_w, psi_x_q, psi_w_w, psi_w_q, psi_q_q = psi_second_order 

print "\nEnd of script!"


