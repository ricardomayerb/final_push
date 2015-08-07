# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 21:09:58 2014

@author: ricardomayerb
"""

import fipir_new
import scipy.linalg

import numpy as np
import sympy

reload(fipir_new)

x_names = ['a', 'C', 'I', 'K', 'L', 'Y', 'zetaa']
x_dates = ['tm1', 't', 'tp1']

w_names = ['wa', 'wzetaa']
w_dates = ['t', 'tp1']

param_names = ['alpha', 'beta', 'delta', 'mua',
               'rhoa', 'nu', 'sigmaa', 'sigmazetaa']

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


xini_az = np.zeros((7))
xini_az_other = np.array([0.6,  2.,  0.5,  0.3, 1.,  0.02,  0.01])
xini_az[0] = xini_az_other[5]
xini_az[1] = xini_az_other[0]
xini_az[2] = xini_az_other[3]
xini_az[3] = xini_az_other[1]
xini_az[4] = xini_az_other[6]
xini_az[5] = xini_az_other[2]
xini_az[6] = xini_az_other[4]
xss_ini_dict = dict(zip(x_in_ss_sym_li, xini_az))
# x_in_ss_sym_li
# Out[27]: [a_ss, C_ss, I_ss, K_ss, zetaa_ss, L_ss, Y_ss]


# this is an alternative to the exec function below
# it creates sympy variables in the global name space
# experimenting to see if it works

sympy.var(param_sym_d.keys())
sympy.var(x_s_d.keys())
sympy.var(w_s_d.keys())

# for k, v in x_s_d.iteritems():
#     exec(k + '= v')

# for k, v in w_s_d.iteritems():
#     exec(k + '= v')

# for k, v in param_sym_d.iteritems():
#     exec(k + '= v')


num_x_state_space = 1
num_s_state_space = 1

s_space_matrices_sym = fipir_new.make_state_space_sym(num_s_state_space,
                                                      num_x_state_space, True)

# print  s_space_matrices_sym.keys()
# print 'A_z state space:', s_space_matrices_sym['A_z']
# print 'C_z state space', s_space_matrices_sym['C_z']
# print 'D_s state space', s_space_matrices_sym['D_s']
# print 'G_s state space', s_space_matrices_sym['G_s']

A_z_sym = s_space_matrices_sym['A_z']
C_z_sym = s_space_matrices_sym['C_z']
D_s_sym = s_space_matrices_sym['D_s']
G_s_sym = s_space_matrices_sym['G_s']


# %%
# ==============================================================================
# ELW 2007 calibration for annual data
# beta = 0.98
# \zeta = 3 (this implies nu = 1/4, right?)
# gamma = 1, this is log utility
# alpha = 0.36
# delta = 0.10
# ==============================================================================

# ==============================================================================
# set numerical values to define an instance
# ==============================================================================
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


state_space_params_usernames_to_values = {'mua': mua_value_true_Toy,
                                          'rhoa': rhoa_value,
                                          'sigmaa': sigmaa_value,
                                          'sigmazetaa': sigmaza_value}


state_space_params_modulenames_to_values = {'mu_0': mua_value_true_Toy,
                                            'rho_0': rhoa_value,
                                            'sigma_signal_0': sigmaa_value,
                                            'sigma_state_0': sigmaza_value}


# this is ELW's \phi, signal-to-noise ratio
ratio_var = (sigmaza_value/sigmaa_value)**2

SNR_value = 10
sigmaza_value_from_SNR = np.sqrt(SNR_value)*sigmaa_value

oneratphi = (1 - rhoa_value**2 - ratio_var)

sskg_bden = 2 - oneratphi + np.sqrt(oneratphi + 4*ratio_var)

sskg = 1 - 2.0/sskg_bden

print '\n'
print 'sskg=', sskg

# %%


#
# ==============================================================================
# define state space matricex, toy model, numeric
# ==============================================================================


A_z_num, C_z_num, D_s_num, G_s_num = fipir_new.numpy_state_space_matrices(
    A_z_sym, C_z_sym, D_s_sym, G_s_sym,
    state_space_params_modulenames_to_values)

# alternatively
A_z_num2, C_z_num2, D_s_num2, G_s_num2 = fipir_new.numpy_state_space_matrices(
    A_z_sym, C_z_sym, D_s_sym, G_s_sym,
    state_space_params_usernames_to_values,
    user_names=True, mod2user_dic=param_names_module_to_user)

Aelw = np.array([[1, 0], [0, rhoa_value]])
Celw = np.array([[0, 0], [0, sigmaza_value]])
# Delw = np.array([[1.0, rhoavalue]])
Delw = np.array([[1.0, 1.0]])
Gelw = np.array([[sigmaa_value, sigmaza_value]])


rhoa_sym = sympy.Symbol('rho_a', positive=True)

# Ds_elw = sympy.Matrix([[1, rhoa_sym]])
Ds_elw = sympy.Matrix([[1, 1]])

stsp_sdict = {'D': Ds_elw}


tech_dict = {'alpha': alpha_value, 'delta': delta_value}
prefs_dict = {'beta': beta_value, 'nu': nu_value}


prod_dict = {'rhoa': rhoa_value, 'muatrue': mua_value_true_Toy}

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

ss_elw = fipir_new.SignalStateSpace(A_num=A_z_num, C_num=C_z_num,
                                    D_num=D_s_num, G_num=G_s_num,
                                    A_sym=A_z_sym, C_sym=C_z_sym,
                                    D_sym=D_s_sym, G_sym=G_s_sym)


# ==============================================================================
#
# Define full information equilibrum conditions
# ==============================================================================


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

az_dict = {at: mua + zetaat + sigmaa*wat, atp1: mua + zetaatp1 + sigmaa*watp1}
glist_cikly_az = eq_conditions_toy_FI_CIKLY_az()


def eq_conditions_toy_FI_CIKLY_az_infodate_contK():

    It_expr = sympy.exp(at) * Kt - (1-delta)*Ktm1
    Yt_expr = sympy.exp((1-alpha)*at) * Ktm1**(alpha) * Lt**(1-alpha)
    Ytp1_expr = sympy.exp((1-alpha)*atp1) * Kt**(alpha) * Ltp1**(1-alpha)

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

glist_cikly_az_info_date = eq_conditions_toy_FI_CIKLY_az_infodate_contK()


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

# %%
gxtp1, gxt, gxtm1, gwtp1, gwpt, gq = dfi_n

d_second = dse_n

nx = psi_x.shape[1]
nw = psi_w.shape[1]

gxtp1xtp1 = d_second[0]
gxtp1xt = d_second[1]
gxtp1xtm1 = d_second[2]
gxtp1wtp1 = d_second[3]
gxtp1wt = d_second[4]
gxtp1q = d_second[5]

gxtxtp1 = d_second[6]
gxtxt = d_second[7]
gxtxtm1 = d_second[8]
gxtwtp1 = d_second[9]
gxtwt = d_second[10]
gxtq = d_second[11]

gxtm1xtp1 = d_second[12]
gxtm1xt = d_second[13]
gxtm1xtm1 = d_second[14]
gxtm1wtp1 = d_second[15]
gxtm1wt = d_second[16]
gxtm1q = d_second[17]

gwtp1xtp1 = d_second[18]
gwtp1xt = d_second[19]
gwtp1xtm1 = d_second[20]
gwtp1wtp1 = d_second[21]
gwtp1wt = d_second[22]
gwtp1q = d_second[23]

gwtxtp1 = d_second[24]
gwtxt = d_second[25]
gwtxtm1 = d_second[26]
gwtwtp1 = d_second[27]
gwtwt = d_second[28]
gwtq = d_second[29]

gqxtp1 = d_second[30]
gqxt = d_second[31]
gqxtm1 = d_second[32]
gqwtp1 = d_second[33]
gqwt = d_second[34]
gqq = d_second[35]

I_n = np.eye(nx)
I_n_dot_j = [I_n[:, j].reshape((nx, 1)) for j in range(nx)]
psiwkronIndotj_list = [np.kron(psi_w, c) for c in I_n_dot_j]
psiwkronIndotj = np.concatenate(psiwkronIndotj_list, axis=1)
psiwkronIk = np.kron(psi_w, np.eye(nw))
psixkronIn = np.kron(psi_x, np.eye(nx))
psixkronIk = np.kron(psi_x, np.eye(nw))
psiqkronIn = np.kron(psi_q, np.eye(nx))
psiqkronIk = np.kron(psi_q, np.eye(nw))

Inkronpsix = np.kron(np.eye(nx), psi_x)
Inkronpsiw = np.kron(np.eye(nx), psi_w)
Inkronpsiq = np.kron(np.eye(nx), psi_q)
Inkronpsixsquared = np.kron(np.eye(nx), np.dot(psi_x, psi_x))
Ikkronpsix = np.kron(np.eye(nw), psi_x)

psixkronpsix = np.kron(psi_x, psi_x)
psixkronpsiw = np.kron(psi_x, psi_w)
psiwkronpsix = np.kron(psi_w, psi_x)
psixkronpsiq = np.kron(psi_x, psi_q)
psiqkronpsix = np.kron(psi_q, psi_x)
psiwkronpsiq = np.kron(psi_w, psi_q)
psiqkronpsiw = np.kron(psi_q, psi_w)
psiwkronpsiw = np.kron(psi_w, psi_w)

print '\npsixkronIk.shape:', psixkronIk.shape
print '\npsixkronIn.shape:', psixkronIn.shape
print '\nInkronpsix.shape:', Inkronpsix.shape
print '\nInkronpsixsquared.shape:', Inkronpsixsquared.shape
print '\nIkkronpsix.shape:', Ikkronpsix.shape
print '\npsixkronpsix.shape:', psixkronpsix.shape

# equation for psi_xx
# A psi_xx + gxtp1 psi_xx B + C
# A = gxt + gxtp1 psi_x
# B = (psi_x kron psi_x)
# C =big constant
# vectorized solution:
# [(I_nn kron A) + (B' kron gxtp1)] vec(psi_xx) = -vec(C)

A_for_psixx = gxt + np.dot(gxtp1, psi_x)
B_for_psixx = np.kron(psi_x, psi_x)

print '\nA_for_psixx.shape:', A_for_psixx.shape
print '\nB_for_psixx.shape:', B_for_psixx.shape

C_for_psixx_1 = gxtm1xtm1 + 2*np.dot(gxtm1xt, Inkronpsix)
print '\nC_for_psixx_1.shape:', C_for_psixx_1.shape
C_for_psixx_2 = gxtxt + 2*np.dot(gxtxtp1, Inkronpsix) \
    + np.dot(gxtp1xtp1, psixkronpsix)
print '\nC_for_psixx_2.shape:', C_for_psixx_2.shape
C_for_psixx_3 = 2*np.dot(gxtm1xtp1, Inkronpsixsquared)
print '\nC_for_psixx_3.shape:', C_for_psixx_3.shape
C_for_psixx = C_for_psixx_1 + np.dot(C_for_psixx_2, psixkronpsix) \
    + C_for_psixx_3
print '\nC_for_psixx.shape:', C_for_psixx.shape
leftmat = np.kron(np.eye(nx*nx), A_for_psixx) + np.kron(B_for_psixx.T, gxtp1)
rightmat = - C_for_psixx.T.flatten()  # one dimensional array
rightmat2d = - C_for_psixx.T.reshape(-1, 1)  # two dimensional column array

print '\nleftmat.shape:', leftmat.shape
print '\nrightmat.shape:', rightmat.shape
print '\nrightmat2d.shape:', rightmat2d.shape

invleftmat = scipy.linalg.inv(leftmat)

vec_psixx_sol = np.dot(invleftmat, rightmat)
vec_psixx_sol2d = np.dot(invleftmat, rightmat2d)

vec_psi_x_x_bysolve = scipy.linalg.solve(leftmat, rightmat)  # twice as fast


psi_x_x = vec_psixx_sol.reshape((nx, nx*nx), order='F')
psi_x_x_2d = vec_psixx_sol2d.reshape((nx, nx*nx), order='F')
print '\npsi_x_x.shape:', psi_x_x.shape
print '\npsi_x_x_2d.shape:', psi_x_x_2d.shape


# equation for psi_x_w
# 0= 2*Gammax2*psi_xw + 2*gtm1wt + Gammaxw*(psi_x kron Ik) +
# Gammaxtm1xt*(In kron psi_x)

Gamma_xtp1_2 = gxtp1
Gamma_xtp1_xtp1 = gxtp1xtp1
Gamma_xtp1_wtp1 = 2*gxtp1wtp1 + np.dot(Gamma_xtp1_xtp1, psiwkronIndotj)
Gamma_xtp1_wt = 2*gxtp1wt
Gamma_xtp1_q = 2*gxtp1q + np.dot(Gamma_xtp1_xtp1, psiqkronIn)

Gamma_xt_2 = gxt + np.dot(Gamma_xtp1_2, psi_x)
Gamma_xt_xtp1 = 2*gxtxtp1 + np.dot(Gamma_xtp1_xtp1, psixkronIn)
Gamma_xt_xt = gxtxt + \
    np.dot(Gamma_xtp1_2, psi_x_x) + np.dot(Gamma_xt_xtp1, Inkronpsix)
Gamma_xt_wt = 2*gxtwt + \
    np.dot(Gamma_xtp1_wt, psixkronIk) + np.dot(Gamma_xt_xt, psiwkronIndotj)


Gamma_xtm1_xtp1 = 2*gxtm1xtp1
Gamma_xtm1_xt = 2*gxtm1xt + \
    np.dot(Gamma_xt_xt, psixkronIn) + np.dot(Gamma_xtm1_xtp1, Inkronpsix)

Gamma_wtp1_wt = np.dot(Gamma_xtp1_wt, psiwkronIk)


A_for_psi_x_w = 2*Gamma_xt_2
b_for_psi_x_w = - 2*gxtm1wt + \
    np.dot(Gamma_xt_wt, psixkronIk) + np.dot(Gamma_xtm1_xt, Inkronpsiw)
psi_x_w = np.dot(scipy.linalg.inv(A_for_psi_x_w), b_for_psi_x_w)
psi_x_w_bysolve = scipy.linalg.solve(A_for_psi_x_w, b_for_psi_x_w)


Gamma_xt_wtp1 = 2*gxtwtp1 + 2 * np.dot(Gamma_xtp1_2, psi_x_w) + np.dot(
    Gamma_xtp1_wtp1, psixkronIk) + np.dot(Gamma_xt_xtp1, Inkronpsiw)
Gamma_xtm1_wtp1 = 2*gxtm1wtp1 + \
    np.dot(Gamma_xt_wtp1, psixkronIk) + np.dot(Gamma_xtm1_xtp1, Inkronpsiw)

Gamma_wt_wtp1 = 2*gwtwtp1 + np.dot(Gamma_xt_wtp1, psiwkronIk)

# equation for psi_x_w
# Gammax2*psi_xw + gww +  Gammaxw*(psi_w kron Ik)
A_for_psi_w_w = Gamma_xt_2
b_for_psi_w_w = gwtwt + np.dot(Gamma_xt_wt, psiwkronIk)
psi_w_w = np.dot(scipy.linalg.inv(A_for_psi_w_w), b_for_psi_w_w)
psi_w_w_bysolve = scipy.linalg.solve(A_for_psi_w_w, b_for_psi_w_w)


# Compute Vxx
utility_sym_mat = sympy.Matrix([elw_fi.utility])
du_dx = utility_sym_mat.jacobian([elw_fi.xvar_t_sym])
du_dx_nopar = [u.subs(elw_fi.par_to_values_dict) for u in list(du_dx)]
du_dx_at_ss = [u.subs(elw_fi.normal_and_0_to_ss).subs(
    elw_fi.ss_solutions_dict) for u in du_dx_nopar]
du_dx_at_ss = elw_fi.matrix2numpyfloat(sympy.Matrix(du_dx_at_ss))

beta = elw_fi.beta.subs(elw_fi.par_to_values_dict)
Ibetaphi = np.eye(nx) - beta*psi_x
invIbetaphi = scipy.linalg.inv(Ibetaphi)
Vx = np.dot(du_dx_at_ss.T, invIbetaphi)

du_dx = utility_sym_mat.jacobian([elw_fi.xvar_t_sym])
du_dx_dx = du_dx.jacobian(elw_fi.xvar_t_sym)
du_dx_dx_nopar = du_dx_dx.subs(elw_fi.par_to_values_dict)
du_dx_dx_at_ss = du_dx_dx_nopar.subs(
    elw_fi.normal_and_0_to_ss).subs(elw_fi.ss_solutions_dict)
du_dx_dx_at_ss = elw_fi.matrix2numpyfloat(du_dx_dx_at_ss)
# now, vectorize and then transpose the hessian. Becomes 1 \times nx^2 vector
du_dx_dx_at_ss = du_dx_dx_at_ss.T.reshape(-1, 1).T


Vxx_term_1 = du_dx_dx_at_ss + beta * np.dot(Vx, psi_x_x)
inv_of_Vxx_term_2 = np.eye(nx*nx) - beta*psixkronpsix
Vxx = np.dot(Vxx_term_1, scipy.linalg.inv(inv_of_Vxx_term_2))

# Gx
Gx_term_1 = np.dot(gxtp1, psi_w) + gwtp1
Gx_term_2 = 2*np.dot(Vx, psi_x_w) + np.dot(Vxx, psixkronpsiw)
Gx_term_2 = fipir_new.matrix2numpyfloat(Gx_term_2)
# matGx_term_2 = Gx_term_2.reshape(nw,nx)
matGx_term_2 = Gx_term_2.reshape(nw, nx, order='F')
Gx_term_3 = np.dot(Vxx, psiwkronpsix)
Gx_term_3 = fipir_new.matrix2numpyfloat(Gx_term_3)
# matGx_term_3 = Gx_term_3.reshape(nx,nw)
matGx_term_3 = Gx_term_3.reshape(nx, nw, order='F')

theta = elw_fi.theta
# foo = matGx_term_2 + matGx_term_3.T
Gx = np.dot(Gx_term_1, matGx_term_2 + matGx_term_3.T)/theta


# Equation (50) bearing on determintation of psi_x_q
# coe1 psixq + coe2 psixq coe3 + constant terms = 0
# convert it as
# (I kron coe1) vec(psixq) + (coe3' kron coe2) vec(psixq) = -vec(constant terms)
# [(I kron coe1) + (coe3' kron coe2)] vec(psixq)  = -vec(constant terms)
# Solve it as Ax = b


coe1 = 2*Gamma_xt_2  # is 7x7
coe2 = 2*Gamma_xtp1_2  # is 7x7
coe3 = psi_x  # is 7x7


# now some constant terms:
eq50cons_1 = 2*gxtm1q + \
    np.dot(Gamma_xtm1_xtp1, Inkronpsiq) + np.dot(Gamma_xtm1_xt, Inkronpsiq)

eq50cons_2_a = 2*gxtq + 2*np.dot(Gamma_xtp1_q, psi_x)
eq50cons_2_b = np.dot(Gamma_xt_xtp1, Inkronpsiq) + \
    np.dot(Gamma_xt_xt, psiqkronIn)
eq50cons_2 = np.dot(eq50cons_2_a+eq50cons_2_b, psi_x)

E_w_dist = - np.dot(Vx, psi_w).T/theta
eq50cons_3_list = [np.dot(E_w_dist.T, Gamma_xtm1_wtp1[i, :].reshape(
    (nw, nx), order='F')) for i in range(nx)]

eq50cons_3 = np.concatenate(eq50cons_3_list)

eq50cons = eq50cons_1 + eq50cons_2 + eq50cons_3

cons_for_psixq = eq50cons + np.dot(Gx, psi_x)
# (coe3' kron coe2) is 49x49, then the I in (I kron coe1) must be 7x7
A_for_vecpsixq = np.kron(np.eye(nx), coe1) + np.kron(coe3.T, coe2)
b_for_vecpsixq = -cons_for_psixq.reshape((-1, 1))
vec_psixq = scipy.linalg.solve(A_for_vecpsixq, b_for_vecpsixq)
psi_x_q = vec_psixq.reshape((nx, nx), order='F')

# equation for psi_w_q:
# A psiwq + constants = 0
# A is 2*Gamma_{x2}
# psiwq is a matrix and so it is constants. We could vectorize the equation as
# (I kron A) vec(psiwq) = - vec(constants)

eq51_coe = 2 * Gamma_xt_2
eq51_cons_1 = 2*gwtq + \
    np.dot(Gamma_xtp1_wt, psiqkronIk) + np.dot(Gamma_xt_wt, psiqkronIk)

Gamma_xt_q_1of2 = 2*gxtq + 2 * \
    np.dot(Gamma_xtp1_2, psi_x_q) + np.dot(Gamma_xtp1_q, psi_x)
Gamma_xt_q_2of2 = np.dot(
    Gamma_xt_xtp1, Inkronpsiq)+np.dot(Gamma_xt_xt, psiqkronIn)
Gamma_xt_q = Gamma_xt_q_1of2 + Gamma_xt_q_2of2

eq51_cons_2 = np.dot(Gamma_xt_q, psi_w)


eq51_cons_3a_list = [np.dot(E_w_dist.T, Gamma_wtp1_wt[i, :].reshape(
    (nw, nw), order='F')) for i in range(nx)]
eq51_cons_3a = np.concatenate(eq51_cons_3a_list)

eq51_cons_3b_list = [np.dot(E_w_dist.T, Gamma_wt_wtp1[i, :].reshape(
    (nw, nw), order='F')) for i in range(nx)]
eq51_cons_3b = np.concatenate(eq51_cons_3b_list)

eq51_cons_3 = eq51_cons_3a + eq51_cons_3b

eq51_cons = eq51_cons_1 + eq51_cons_2 + eq51_cons_3
cons_for_psiwq = eq51_cons + np.dot(Gx, psi_w)

A_for_vecpsiwq = np.kron(np.eye(nw, nw), eq51_coe)
b_for_vecpsiwq = -cons_for_psiwq.reshape((-1, 1))
vec_psiwq = scipy.linalg.solve(A_for_vecpsiwq, b_for_vecpsiwq)
psi_w_q = vec_psiwq.reshape((nx, nw), order='F')


# equation for Vxq:
# du_dx =  utility_sym_mat.jacobian([elw_fi.xvar_t_sym])
du_dx_dq = du_dx.jacobian(sympy.Matrix([q]))
du_dx_dq_nopar = du_dx_dq.subs(elw_fi.par_to_values_dict)
du_dx_dq_at_ss = du_dx_dq_nopar.subs(
    elw_fi.normal_and_0_to_ss).subs(elw_fi.ss_solutions_dict)
du_dx_dq_at_ss = elw_fi.matrix2numpyfloat(du_dx_dq_at_ss)
# now, vectorize and then transpose the hessian. Becomes 1 \times nx^2 vector
du_dx_dq_at_ss = du_dx_dq_at_ss.T.reshape(-1, 1).T
uxq = du_dx_dq_at_ss

Vxiq_term1 = beta*np.dot(Vx, psi_x_q)+0.5 * \
    np.dot(Vxx, psixkronpsiq+psiqkronpsix)

Vxq_term2a = 2*np.dot(Vx, psi_x_w) + np.dot(Vxx, psixkronpsiw)
Vxq_term2a = Vxq_term2a.reshape((nw, nx), order='F')
Vxq_term2b = np.dot(Vxx, psiwkronpsix)
Vxq_term2b = Vxq_term2b.reshape((nx, nw), order='F')
Vxq_term2c = np.dot(Vx, psi_w)
Vxq_term2 = -beta*0.5*np.dot(Vxq_term2c, (Vxq_term2a + Vxq_term2b.T))/theta
Vxq_rhs = uxq + Vxiq_term1 + Vxq_term2
Vxq_lhs = np.eye(nx) - beta*psi_x
# Vxq = np.dot(Vxq_rhs, scipy.linalg.inv(Vxq_lhs))
Vxq = scipy.linalg.solve(Vxq_lhs.T, Vxq_rhs.T).T  # solving A.T x.T = b.T


# equation for psi_q_q
Eq52_coe = Gamma_xtp1_2 + Gamma_xt_2
Eq52_cons_1 = gqq + np.dot(Gamma_xtp1_q + Gamma_xt_q, psi_q)

Gamma_wtp1_q_a = 2*gwtp1q + 2 * \
    np.dot(Gamma_xtp1_2, psi_w_q) + np.dot(Gamma_xtp1_wtp1, psiqkronIk)
Gamma_wtp1_q_b = np.dot(
    Gamma_xtp1_q, psi_w) + np.dot(Gamma_xt_wtp1, psiqkronIk)
Gamma_wtp1_q = Gamma_wtp1_q_a + Gamma_wtp1_q_b

Eq52_cons_2a = np.dot(Gamma_wtp1_q, E_w_dist)

Gamma_wtp1_wtp1 = gwtp1wtp1 + \
    np.dot(Gamma_xtp1_2, psi_w_w)+np.dot(Gamma_xtp1_wtp1, psiwkronIk)

E_wkronw_dist = np.eye(
    nw) + np.dot(np.dot(Vx, psi_w).T, np.dot(Vx, psi_w))/(theta**2)
E_wkronw_dist = E_wkronw_dist.reshape((-1, 1))
Eq52_cons_2b = np.dot(Gamma_wtp1_wtp1, E_wkronw_dist)
Eq52_cons_2 = Eq52_cons_2a + Eq52_cons_2b
Eq52_cons = Eq52_cons_1 + Eq52_cons_2


cons_psiqq_a = (np.dot(gxtp1, psi_w) + gwtp1)/theta

cons_psiqq_b = 2*np.dot(Vx, psi_w_q) + np.dot(Vxx,
                                              (psiwkronpsiq+psiqkronpsiw)) + 2*np.dot(Vxq, psi_w)
cons_psiqq_ab = np.dot(cons_psiqq_a, cons_psiqq_b.T)

cons_psiqq_c_1 = (
    np.dot(Vx, psi_w_w) + np.dot(Vxx, psiwkronpsiw)).reshape((nw, nw), order='F')
cons_psiqq_c = np.dot(cons_psiqq_c_1.T, E_w_dist)
cons_psiqq_ac = np.dot(cons_psiqq_a, cons_psiqq_c)

cons_psiqq_d = np.dot(cons_psiqq_c_1, E_w_dist)
cons_psiqq_ad = np.dot(cons_psiqq_a, cons_psiqq_d)

terms_in_Eq56 = np.dot(Gx, psi_q) - cons_psiqq_ab - \
    cons_psiqq_ac - cons_psiqq_ad
# terms_in_Eq56 =  np.dot(Gx, psi_q) + cons_psiqq_ab - cons_psiqq_ac - cons_psiqq_ad

cons_for_psiqq = Eq52_cons + terms_in_Eq56

psi_q_q = scipy.linalg.solve(Eq52_coe, -cons_for_psiqq)


print "\nEnd of script!"
# %%


# %%


# elw_fi.get_second_order_approx_coeff_fi(psi_x, psi_w, d_first=dfi_n,
#                                        d_second=dse_n)


# elw_fi = fipir_new.FullInfoModel(ss_elw, x_names, w_names, param_names, xw_sym_dicts=xw_sym_dicts ,
#                                 par_to_values_dict=all_param_values_dict,
#                                 eq_conditions=glist_cikly_az,
#                                 utility=utility_elw,
#                                 xss_ini_dict=xss_ini_dict)
#
#
#
# elw_ciklyaz_fi = fipir_new.FullInfoModel(ss_elw, x_names, w_names, param_names, xw_sym_dicts=xw_sym_dicts ,
#                                 par_to_values_dict=all_param_values_dict,
#                                 eq_conditions=glist_cikly_az,
#                                 utility=utility_elw,
#                                 xss_ini_dict=xss_ini_dict)
#
# elw_ciklyz_fi = fipir_new.FullInfoModel(ss_elw, x_names, w_names, param_names, xw_sym_dicts=xw_sym_dicts_z ,
#                                 par_to_values_dict=all_param_values_dict,
#                                 eq_conditions=glist_cikly_z,
#                                 utility=utility_elw,
#                                 xss_ini_dict=xss_ini_dict_z)
#
# elw_cklaz_fi = fipir_new.FullInfoModel(ss_elw, x_names, w_names, param_names, xw_sym_dicts=xw_sym_dicts_ckl_az ,
#                                 par_to_values_dict=all_param_values_dict,
#                                 eq_conditions=glist_ckl_az,
#                                 utility=utility_elw,
#                                 xss_ini_dict=xss_ini_dict_ckl_az)
#
# elw_cklz_fi = fipir_new.FullInfoModel(ss_elw, x_names, w_names, param_names, xw_sym_dicts=xw_sym_dicts_ckl_z ,
#                                 par_to_values_dict=all_param_values_dict,
#                                 eq_conditions=glist_ckl_z,
#                                 utility=utility_elw,
#                                 xss_ini_dict=xss_ini_dict_ckl_z)
#
#
#psi_x_ciklyaz, psi_w_ciklyaz, psi_q_ciklyaz =  elw_ciklyaz_fi.get_first_order_approx_coeff_fi()
#psi_x_ciklyz, psi_w_ciklyz, psi_q_ciklyz =  elw_ciklyz_fi.get_first_order_approx_coeff_fi()
#psi_x_cklaz, psi_w_cklaz, psi_q_cklaz =  elw_cklaz_fi.get_first_order_approx_coeff_fi()
#psi_x_cklz, psi_w_cklz, psi_q_cklz =  elw_cklz_fi.get_first_order_approx_coeff_fi()
#
