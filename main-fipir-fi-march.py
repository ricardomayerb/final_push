# -*- coding: utf-8 -*-
"""
Created on March 12 2015

@author: ricardomayerb
"""
import sympy
import fipir_new

import numpy as np

# reload(fipir_new)

x_names = ['a', 'C', 'I', 'K', 'L', 'Y', 'zetaa']
x_dates = ['tm1', 't', 'tp1']

w_names = ['wa', 'wzetaa']
w_dates = ['t', 'tp1']

param_names = ['alpha', 'beta', 'delta', 'mua', 'rhoa',
               'nu', 'sigmaa', 'sigmazetaa']

param_names_user_to_module = {'rhoa': 'rho_0', 'mua': 'mu_0',
                              'sigmaa': 'sigma_signal_0',
                              'sigmazetaa': 'sigma_state_0'}

param_names_module_to_user = {'rho_0': 'rhoa', 'mu_0': 'mua',
                              'sigma_signal_0': 'sigmaa',
                              'sigma_state_0': 'sigmazetaa'}


xw_sym_dicts = fipir_new.set_x_w_sym_dicts(x_names, w_names)
x_s_d, w_s_d = xw_sym_dicts


param_sym_d = fipir_new.set_param_sym_dict(param_names)
x_in_ss_sym_d = {st: x_s_d[st] for st in x_s_d.keys() if 'ss' in st}
x_in_ss_sym_li = x_in_ss_sym_d.values()


# xini_az_other = np.array([ 0.6 ,  2.  ,  0.5 ,  0.3 , 1.  ,  0.02,  0.01])
xini_az_cand1 = np.array([0.02, 0.6, 0.3, 2.0, 0.01, 0.5, 1.0])

x_ini = xini_az_cand1

xss_ini_dict = dict(zip(x_in_ss_sym_li, x_ini))

for k, v in x_s_d.iteritems():
    exec(k + '= v')

for k, v in w_s_d.iteritems():
    exec(k + '= v')

for k, v in param_sym_d.iteritems():
    exec(k + '= v')

num_x_state_space = 1
num_s_state_space = 1

s_space_matrices_sym = fipir_new.make_state_space_sym(num_s_state_space,
                                                      num_x_state_space,
                                                      True)

A_z_sym = s_space_matrices_sym['A_z']
C_z_sym = s_space_matrices_sym['C_z']
D_s_sym = s_space_matrices_sym['D_s']
G_s_sym = s_space_matrices_sym['G_s']


# ==============================================================================
# set numerical values to define an instance
# ==============================================================================
alpha_value = 0.36
beta_value = 0.98
delta_value = 0.1
nu_value = 0.29
mua_value_true = 0.03
rhoa_value = 0.97  # 0.01087417, 0.00213658
sigmaza_value = 0.01 * 1
sigmaa_value = 0.01 * 7.5  # sskg = sskg= 0.128156077839
mua_value_SSFI = mua_value_true
mua_tm = mua_value_true

pref_tech_names_to_values = {'alpha': alpha_value, 'beta': beta_value,
                             'delta': delta_value, 'nu': nu_value}

state_space_params_usernames_to_values = {'mua': mua_value_true,
                                          'rhoa': rhoa_value,
                                          'sigmaa': sigmaa_value,
                                          'sigmazetaa': sigmaza_value}

state_space_params_modulenames_to_values = {'mu_0': mua_value_true,
                                            'rho_0': rhoa_value,
                                            'sigma_signal_0': sigmaa_value,
                                            'sigma_state_0': sigmaza_value}

# this is ELW's \phi, signal-to-noise ratio
ratio_var = (sigmaza_value / sigmaa_value) ** 2

SNR_value = 10
sigmaza_value_from_SNR = np.sqrt(SNR_value) * sigmaa_value
oneratphi = (1 - rhoa_value ** 2 - ratio_var)
sskg_bden = 2 - oneratphi + np.sqrt(oneratphi + 4 * ratio_var)
sskg = 1 - 2.0 / sskg_bden

print '\n'
print 'sskg=', sskg
# define state space matricex, toy model, numeric
# ==============================================================================


A_z_num, C_z_num, D_s_num, G_s_num = fipir_new.numpy_state_space_matrices(
    A_z_sym, C_z_sym, D_s_sym, G_s_sym,
    state_space_params_modulenames_to_values)

Aelw = np.array([[1, 0], [0, rhoa_value]])
Celw = np.array([[0, 0], [0, sigmaza_value]])
# Delw = np.array([[1.0, rhoavalue]])
Delw = np.array([[1.0, 1.0]])
Gelw = np.array([[sigmaa_value, sigmaza_value]])

tech_dict = {'alpha': alpha_value, 'delta': delta_value}
prefs_dict = {'beta': beta_value, 'nu': nu_value}


prod_dict = {'rhoa': rhoa_value, 'muatrue': mua_value_true}

# tech_prefs_prod_dict = dict(prefs_dict, **dict(tech_dict, **prod_dict))

tech_prefs_prod_dict = {}
tech_prefs_prod_dict.update(tech_dict)
tech_prefs_prod_dict.update(prefs_dict)
tech_prefs_prod_dict.update(prod_dict)

all_param_values_dict = {}
all_param_values_dict.update(pref_tech_names_to_values)
all_param_values_dict.update(state_space_params_usernames_to_values)
all_param_values_dict.update(state_space_params_modulenames_to_values)


ss_elw_novals = fipir_new.SignalStateSpace(A_sym=A_z_sym, C_sym=C_z_sym,
                                           D_sym=D_s_sym, G_sym=G_s_sym)

ss_elw_dic_vals = fipir_new.SignalStateSpace(A_sym=A_z_sym, C_sym=C_z_sym,
                                             D_sym=D_s_sym, G_sym=G_s_sym,
                                             param_val_dic=all_param_values_dict)

ss_elw_dic_vals_names_dic = fipir_new.SignalStateSpace(A_sym=A_z_sym,
                                                       C_sym=C_z_sym,
                                                       D_sym=D_s_sym,
                                                       G_sym=G_s_sym,
                                                       param_val_dic=all_param_values_dict,
                                                       user_names_dic=param_names_module_to_user)

ss_elw = fipir_new.SignalStateSpace(A_num=A_z_num, C_num=C_z_num,D_num=D_s_num,
                                    G_num=G_s_num, A_sym=A_z_sym, C_sym=C_z_sym,
                                    D_sym=D_s_sym, G_sym=G_s_sym)


def eq_conditions_toy_FI_CIKLY_az():
    
    It_expr = sympy.exp(at) * Ktp1 - (1-delta)*Kt
    Yt_expr = sympy.exp((1-alpha)*at)  * Kt**(alpha) * Lt**(1-alpha) 
    Ytp1_expr = sympy.exp((1-alpha)*atp1)  * Ktp1**(alpha) * Ltp1**(1-alpha)
  
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


def eq_conditions_toy_FI_CIKLY_az_infodate_contK():

    It_expr = sympy.exp(at) * Kt - (1-delta)*Ktm1
    Yt_expr = sympy.exp((1-alpha)*at) * Ktm1**(alpha) * Lt**(1-alpha)
    Ytp1_expr = sympy.exp((1-alpha)*atp1)  * Kt**(alpha) * Ltp1**(1-alpha)
  
    at_expr = mua + zetaat + sigmaa*wat
    atp1_expr = mua + zetaatp1 + sigmaa*watp1
    
    zetaat_expr = rhoa*zetaatm1 + sigmazetaa*wzetaat    
    
    g1 = beta*sympy.exp(-at) * (Ct/Ctp1) * \
         (alpha * Ytp1/Kt + (1-delta)) - 1
            
    g2 = Yt - It - Ct 
    
    g3 = nu*(1-alpha)*Yt/(Ct*(1-nu)) - Lt/(1-Lt)
    
    g4 = zetaat - zetaat_expr
    
    g5 = at - at_expr
    
    g6 = Yt - Yt_expr
    
    g7 = It - It_expr        
    
    glist = [g1, g2, g3, g4, g5, g6, g7]    
    return glist

my_invout_dict = {Yt:sympy.exp((1-alpha)*at)  * Ktm1**(alpha) * Lt**(1-alpha),
              Ytp1:sympy.exp((1-alpha)*atp1) * Kt**(alpha) * Ltp1**(1-alpha),
                It:sympy.exp(at) * Kt - (1-delta)*Ktm1}
                
my_glist_cikly_az = eq_conditions_toy_FI_CIKLY_az_infodate_contK()

utility_elw = nu * sympy.log(Ct) + (1-nu) * sympy.log(Lt)                                

elw_fi = fipir_new.FullInfoModel(ss_elw, x_names, w_names, param_names,
                                 xw_sym_dicts=xw_sym_dicts,
                                 par_to_values_dict=all_param_values_dict,
                                 eq_conditions=glist_cikly_az,
                                 utility=utility_elw,
                                 xss_ini_dict=xss_ini_dict)

dfi_unev, dse_unev = elw_fi.d1d2_g_x_w_unevaluated()

fun1n, fun2n, vn = elw_fi.make_numpy_fns_of_d1d2xw(dfi_unev, dse_unev)
fun1t, fun2t, vt = elw_fi.make_theano_fns_of_d1d2xw(dfi_unev, dse_unev)

dfi_n, dse_n = elw_fi.get_evaluated_dgdxw12(mod='numpy')
dfi_t, dse_t = elw_fi.get_evaluated_dgdxw12(mod='theano')

psi_x, psi_w, psi_q = elw_fi.get_first_order_approx_coeff_fi()


