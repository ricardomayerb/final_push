import fipiro
import fipiro.utils as u
import sympy
from sympy.utilities.lambdify import lambdify
from math import exp
import numpy as np
reload(fipiro.utils)


x_names = ['a', 'C', 'I', 'K', 'L', 'Y', 'zetaa']
x_names_uh = ['adm', 'bdm', 'C', 'I', 'K', 'L', 'Y', 'zetaa']
x_dates = ['tm1', 't', 'tp1']
w_names = ['wa', 'wzetaa']
w_dates = ['t', 'tp1']
param_names = ['alpha', 'beta', 'delta', 'mua',
               'rhoa', 'nu', 'sigmaa', 'sigmazetaa']
all_names = {
    'x_names': x_names, 'w_names': w_names, 'param_names': param_names}
#
# param_names_module_to_user = {'rho_0': 'rhoa', 'mu_0': 'mua',
#                               'sigma_signal_0': 'sigmaa',
#                               'sigma_state_0': 'sigmazetaa'}

xxsswp_sym_d = u.make_x_w_param_sym_dicts(x_names, w_names, param_names)
x_s_d, x_in_ss_sym_d, w_s_d, param_sym_d = xxsswp_sym_d
xss_ini_dict = {x_in_ss_sym_d['Y_ss']: 1.0,
                x_in_ss_sym_d['K_ss']: 2.0,
                x_in_ss_sym_d['zetaa_ss']: 0.01,
                x_in_ss_sym_d['a_ss']: 0.01,
                x_in_ss_sym_d['I_ss']: 0.3,
                x_in_ss_sym_d['C_ss']: 0.6,
                x_in_ss_sym_d['L_ss']: 0.5}

xxsswp_sym_d_uh = u.make_x_w_param_sym_dicts(x_names_uh, w_names, param_names)
x_s_d_uh, x_in_ss_sym_d_uh, w_s_d, param_sym_d = xxsswp_sym_d_uh
xss_ini_dict_uh = {x_in_ss_sym_d_uh['Y_ss']: 1.0,
                   x_in_ss_sym_d_uh['K_ss']: 2.0,
                   x_in_ss_sym_d_uh['zetaa_ss']: 0.01,
                   x_in_ss_sym_d_uh['adm_ss']: 0.01,
                   x_in_ss_sym_d_uh['bdm_ss']: 0.01,
                   x_in_ss_sym_d_uh['I_ss']: 0.3,
                   x_in_ss_sym_d_uh['C_ss']: 0.6,
                   x_in_ss_sym_d_uh['L_ss']: 0.5}

for k, v in x_s_d.iteritems():
    exec(k + '= v')

for k, v in x_s_d_uh.iteritems():
    exec(k + '= v')

for k, v in w_s_d.iteritems():
    exec(k + '= v')

for k, v in param_sym_d.iteritems():
    exec(k + '= v')


def eq_conds_TFP_FI_CIKLY_az_infodating_aleads_sta_all():

    It_expr = Kt - (1-delta) * sympy.exp(-atm1) * Ktm1

    Yt_expr = sympy.exp((1-alpha)*at - alpha*atm1) * Ktm1**(alpha) \
        * Lt**(1-alpha)
    Ytp1_expr = sympy.exp((1-alpha)*atp1 - alpha*at) * Kt**(alpha) \
        * Ltp1**(1-alpha)

    at_expr = mua + zetaatm1 + sigmaa*wat
    atp1_expr = mua + zetaat + sigmaa*watp1

    zetaat_expr = rhoa*zetaatm1 + sigmazetaa*wzetaat

    g1 = beta * (Ct/Ctp1) * (alpha * Ytp1/Kt + (1-delta) * sympy.exp(-at)) - 1

    g2 = nu*(1-alpha)*Yt/(Ct*(1-nu)) - Lt/(1-Lt)

    g3 = Ct + It - Yt

    g4 = It_expr - It

    g5 = Yt_expr - Yt

    g6 = zetaat_expr - zetaat

    g7 = at_expr - at

    glist = [g1, g2, g3, g4, g5, g6, g7]
    invout_dict = {Yt: Yt_expr, Ytp1: Ytp1_expr, It: It_expr}
    az_dict = {at: at_expr, atp1: atp1_expr}
    return glist, az_dict, invout_dict


glist_cikly_az_info, iy1, az1 = \
  eq_conds_TFP_FI_CIKLY_az_infodating_aleads_sta_all()

#
# def eq_conds_TFP_FI_CIKLY_az_infodating_aleads_sta_all_uh_by_hand():
#
#     It_expr = Kt - (1-delta) * sympy.exp(-bdmt - mua) * Ktm1
#
#     Yt_expr = sympy.exp((1-alpha)*(admt + mua) - alpha*(bdmt + mua)) * \
#         Ktm1**(alpha) * Lt**(1-alpha)
#     Ytp1_expr = sympy.exp((1-alpha)*(admtp1 + mua) - alpha*(admt + mua)) * \
#         Kt**(alpha) * Ltp1**(1-alpha)
#
#     admt_expr = zetaatm1 + sigmaa*wat
#     admtp1_expr = zetaat + sigmaa*watp1
#
#     zetaat_expr = rhoa*zetaatm1 + sigmazetaa*wzetaat
#
#     g1 = beta * (Ct/Ctp1) * (alpha * Ytp1/Kt + (1-delta)
#                              * sympy.exp(-admt - mua)) - 1
#
#     g2 = nu*(1-alpha)*Yt/(Ct*(1-nu)) - Lt/(1-Lt)
#
#     g3 = Ct + It - Yt
#
#     g4 = It_expr - It
#
#     g5 = Yt_expr - Yt
#
#     g6 = zetaat_expr - zetaat
#
#     g7 = admt_expr - admt
#
#     g8 = admtm1 - bdmt
#
#     glist = [g1, g2, g3, g4, g5, g6, g7, g8]
#     invout_dict = {Yt: Yt_expr, Ytp1: Ytp1_expr, It: It_expr}
#     az_dict = {admt: admt_expr, admtp1: admtp1_expr}
#     return glist, az_dict, invout_dict
#
# # glist_cikly_az_info_uh, iy1_uh, az1_uh = \
# #   eq_conds_TFP_FI_CIKLY_az_infodating_aleads_sta_all_uh()

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

all_param_values_dict = {}
all_param_values_dict.update(pref_tech_names_to_values)
all_param_values_dict.update(ssp_par_user_to_values)

# bmodel = u.ModelBase(glist_cikly_az_info, x_names=x_names,
#                      w_names=w_names, param_names=param_names,
#                      par_to_values_dict=all_param_values_dict)

# bmodel2 = u.ModelBase(glist_cikly_az_info, x_names=x_names,
#                       w_names=w_names, param_names=param_names,
#                       par_to_values_dict=all_param_values_dict,
#                       vars_initvalues_dict=xss_ini_dict)

non_ex_block_index = (1, 2, 3, 4)
ex_block_index = (0)
# z_block_index = (5, 6)
z_block_index = (5, 6, 7)
uhlig_block_indices = {'expectational_block': ex_block_index,
                       'non_expectational_block': non_ex_block_index,
                       'z_block': z_block_index}
u_x_names = ['K']
u_y_names = ['C', 'I', 'L', 'Y']
u_z_names = ['adm', 'bdm', 'zetaa']
# u_z_names = ['a', 'zetaa']

dic_ori_2_uhlig = {atp1: admtp1+mua, at: admt+mua, atm1:bdmt+mua}
aux_eq1 = [bdmtp1 - admt]

umodel = u.UhligModel(glist_cikly_az_info,
                      x_names=x_names_uh, w_names=w_names,
                      param_names=param_names,
                      block_indices=uhlig_block_indices,
                      u_x_names=u_x_names,
                      u_y_names=u_y_names,
                      u_z_names=u_z_names,
                      aux_eqs=aux_eq1,
                      fwd_shift_idx=(5, 6),
                      u_trans_dict=dic_ori_2_uhlig,
                      par_to_values_dict=all_param_values_dict)


# umodel2 = u.UhligModel(glist_cikly_az_info,
#                        x_names=x_names, w_names=w_names,
#                        param_names=param_names,
#                        block_indices=uhlig_block_indices,
#                        shift_z_forward=True,
#                        u_x_names=u_x_names,
#                        u_y_names=u_y_names,
#                        u_z_names=u_z_names,
#                        par_to_values_dict=all_param_values_dict,
#                        vars_initvalues_dict=xss_ini_dict)
# np.set_printoptions(precision=3)
# print '\nA'
# print umodel.uA_num_ss
# print '\n'
# print umodel.uA_sym_ss
# print '\n'
# print umodel.uA_sym
#
# print '\nB'
# print umodel.uB_num_ss
# print '\n'
# print umodel.uB_sym_ss
# print '\n'
# print umodel.uB_sym
#
# print '\nC'
# print umodel.uC_num_ss
# print '\n'
# print umodel.uC_sym_ss
# print '\n'
# print umodel.uC_sym
#
# print '\nD'
# print umodel.uD_num_ss
# print '\n'
# print umodel.uD_sym_ss
# print '\n'
# print umodel.uD_sym
#
# print '\nF'
# print umodel.uF_num_ss
# print '\n'
# print umodel.uF_sym_ss
# print '\n'
# print umodel.uF_sym
#
# print '\nG'
# print umodel.uG_num_ss
# print '\n'
# print umodel.uG_sym_ss
# print '\n'
# print umodel.uG_sym
#
# print '\nH'
# print umodel.uH_num_ss
# print '\n'
# print umodel.uH_sym_ss
# print '\n'
# print umodel.uH_sym
#
# print '\nJ'
# print umodel.uJ_num_ss
# print '\n'
# print umodel.uJ_sym_ss
# print '\n'
# print umodel.uJ_sym
#
# print '\nK'
# print umodel.uK_num_ss
# print '\n'
# print umodel.uK_sym_ss
# print '\n'
# print umodel.uK_sym
#
# print '\nL'
# print umodel.uL_num_ss
# print '\n'
# print umodel.uL_sym_ss
# print '\n'
# print umodel.uL_sym
#
# print '\nM'
# print umodel.uM_num_ss
# print '\n'
# print umodel.uM_sym_ss
# print '\n'
# print umodel.uM_sym
#
# print '\nN'
# print umodel.uN_num_ss
# print '\n'
# print umodel.uN_sym_ss
# print '\n'
# print umodel.uN_sym
