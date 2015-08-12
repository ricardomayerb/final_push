import fipir_aug
import sympy
import numpy as np
import scipy

reload(fipir_aug)

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

xxsswp_sym_d = fipir_aug.make_x_w_param_sym_dicts(x_names, w_names, param_names)

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

utility_elw = nu * sympy.log(Ct) + (1-nu) * sympy.log(Lt)

all_param_values_dict = {}
all_param_values_dict.update(pref_tech_names_to_values)
all_param_values_dict.update(ssp_par_user_to_values)

fi = fipir_aug.FullInfoModel(ssp, all_names,
                             par_to_values_dict=all_param_values_dict,
                             eq_conditions=glist_cikly_az,
                             utility=utility_elw,
                             xss_ini_dict=xss_ini_dict)
              
dfi_unev, dse_unev = fi.d1d2_g_x_w_unevaluated()

fun1n, fun2n, vn = fi.make_numpy_fns_of_d1d2xw(dfi_unev, dse_unev)



#fun1t, fun2t, vt = fi.make_theano_fns_of_d1d2xw(dfi_unev, dse_unev)


#dfi_n, dse_n = fi.get_evaluated_dgdxw12(mod='numpy')
##dfi_t, dse_t = fi.get_evaluated_dgdxw12(mod='theano')
#
#psi_x, psi_w, psi_q =  fi.get_first_order_approx_coeff_fi()
#
#
#
##%%
#gxtp1, gxt, gxtm1, gwtp1, gwpt, gq = dfi_n
#
#d_second = dse_n
#
#nx =  psi_x.shape[1]
#nw = psi_w.shape[1]
#
#gxtp1xtp1 = d_second[0]
#gxtp1xt = d_second[1]
#gxtp1xtm1 = d_second[2]
#gxtp1wtp1 = d_second[3]
#gxtp1wt = d_second[4]
#gxtp1q = d_second[5]
#
#gxtxtp1 = d_second[6]
#gxtxt = d_second[7]
#gxtxtm1 = d_second[8]
#gxtwtp1 = d_second[9]
#gxtwt = d_second[10]
#gxtq = d_second[11]
#
#gxtm1xtp1 = d_second[12]
#gxtm1xt = d_second[13]
#gxtm1xtm1 = d_second[14]
#gxtm1wtp1 = d_second[15]
#gxtm1wt = d_second[16]
#gxtm1q = d_second[17]
#
#gwtp1xtp1 = d_second[18]
#gwtp1xt = d_second[19]
#gwtp1xtm1 = d_second[20]
#gwtp1wtp1 = d_second[21]
#gwtp1wt = d_second[22]
#gwtp1q = d_second[23]
#
#gwtxtp1 = d_second[24]
#gwtxt = d_second[25]
#gwtxtm1 = d_second[26]
#gwtwtp1 = d_second[27]
#gwtwt = d_second[28]
#gwtq = d_second[29]
#
#gqxtp1 = d_second[30]
#gqxt = d_second[31]
#gqxtm1 = d_second[32]
#gqwtp1 = d_second[33]
#gqwt = d_second[34]
#gqq = d_second[35]
#
#I_n = np.eye(nx)        
#I_n_dot_j =  [I_n[:,j].reshape((nx,1)) for j in range(nx)]
#psiwkronIndotj_list = [np.kron(psi_w, c) for c in I_n_dot_j]
#psiwkronIndotj = np.concatenate(psiwkronIndotj_list, axis=1)
#psiwkronIk = np.kron(psi_w, np.eye(nw))
#psixkronIn = np.kron(psi_x, np.eye(nx))
#psixkronIk = np.kron(psi_x, np.eye(nw))
#psiqkronIn = np.kron(psi_q, np.eye(nx))
#psiqkronIk = np.kron(psi_q, np.eye(nw))
#
#Inkronpsix = np.kron(np.eye(nx), psi_x)
#Inkronpsiw = np.kron(np.eye(nx), psi_w)
#Inkronpsiq = np.kron(np.eye(nx), psi_q)
#Inkronpsixsquared = np.kron(np.eye(nx), np.dot(psi_x, psi_x))
#Ikkronpsix = np.kron(np.eye(nw), psi_x)
#
#psixkronpsix = np.kron(psi_x, psi_x)
#psixkronpsiw = np.kron(psi_x, psi_w)
#psiwkronpsix = np.kron(psi_w, psi_x)
#psixkronpsiq = np.kron(psi_x, psi_q)
#psiqkronpsix = np.kron(psi_q, psi_x)
#psiwkronpsiq = np.kron(psi_w, psi_q)
#psiqkronpsiw = np.kron(psi_q, psi_w)
#psiwkronpsiw = np.kron(psi_w, psi_w)
#
#print '\npsixkronIk.shape:', psixkronIk.shape
#print '\npsixkronIn.shape:', psixkronIn.shape
#print '\nInkronpsix.shape:', Inkronpsix.shape
#print '\nInkronpsixsquared.shape:', Inkronpsixsquared.shape
#print '\nIkkronpsix.shape:', Ikkronpsix.shape
#print '\npsixkronpsix.shape:', psixkronpsix.shape
#
##### equation for psi_xx
## A psi_xx + gxtp1 psi_xx B + C
## A = gxt + gxtp1 psi_x 
## B = (psi_x kron psi_x)
## C =big constant
## vectorized solution:
## [(I_nn kron A) + (B' kron gxtp1)] vec(psi_xx) = -vec(C)
#
#A_for_psixx = gxt + np.dot(gxtp1, psi_x)
#B_for_psixx = np.kron(psi_x, psi_x)
#
#print '\nA_for_psixx.shape:', A_for_psixx.shape
#print '\nB_for_psixx.shape:', B_for_psixx.shape
#
#C_for_psixx_1 = gxtm1xtm1 + 2*np.dot(gxtm1xt,Inkronpsix)
#print '\nC_for_psixx_1.shape:', C_for_psixx_1.shape
#C_for_psixx_2 = gxtxt + 2*np.dot(gxtxtp1, Inkronpsix) \
#             + np.dot(gxtp1xtp1, psixkronpsix) 
#print '\nC_for_psixx_2.shape:', C_for_psixx_2.shape
#C_for_psixx_3 =   2*np.dot(gxtm1xtp1,Inkronpsixsquared)
#print '\nC_for_psixx_3.shape:', C_for_psixx_3.shape
#C_for_psixx = C_for_psixx_1 + np.dot(C_for_psixx_2, psixkronpsix) \
#                 + C_for_psixx_3
#print '\nC_for_psixx.shape:', C_for_psixx.shape
#leftmat = np.kron(np.eye(nx*nx), A_for_psixx) + np.kron(B_for_psixx.T,gxtp1)
#rightmat = - C_for_psixx.T.flatten() # one dimensional array
#rightmat2d = - C_for_psixx.T.reshape(-1,1) # two dimensional column array
#
#print '\nleftmat.shape:', leftmat.shape
#print '\nrightmat.shape:', rightmat.shape
#print '\nrightmat2d.shape:', rightmat2d.shape
#
#invleftmat = scipy.linalg.inv(leftmat)
#
#vec_psixx_sol = np.dot(invleftmat , rightmat)
#vec_psixx_sol2d = np.dot(invleftmat, rightmat2d)
#
#vec_psi_x_x_bysolve = scipy.linalg.solve(leftmat, rightmat) #twice as fast 
#
#
#psi_x_x = vec_psixx_sol.reshape((nx, nx*nx), order='F')
#psi_x_x_2d = vec_psixx_sol2d.reshape((nx, nx*nx), order='F')
#print '\npsi_x_x.shape:', psi_x_x.shape
#print '\npsi_x_x_2d.shape:', psi_x_x_2d.shape
# 
#
##### equation for psi_x_w
##0= 2*Gammax2*psi_xw + 2*gtm1wt + Gammaxw*(psi_x kron Ik) + Gammaxtm1xt*(In kron psi_x)
#
#Gamma_xtp1_2 = gxtp1
#Gamma_xtp1_xtp1 = gxtp1xtp1
#Gamma_xtp1_wtp1 = 2*gxtp1wtp1 + np.dot(Gamma_xtp1_xtp1,psiwkronIndotj)
#Gamma_xtp1_wt = 2*gxtp1wt
#Gamma_xtp1_q = 2*gxtp1q + np.dot(Gamma_xtp1_xtp1, psiqkronIn)
#
#Gamma_xt_2 = gxt + np.dot(Gamma_xtp1_2, psi_x)
#Gamma_xt_xtp1 = 2*gxtxtp1 + np.dot(Gamma_xtp1_xtp1,psixkronIn)
#Gamma_xt_xt = gxtxt + np.dot(Gamma_xtp1_2,psi_x_x) + np.dot(Gamma_xt_xtp1,Inkronpsix)
#Gamma_xt_wt = 2*gxtwt + np.dot(Gamma_xtp1_wt,psixkronIk) + np.dot(Gamma_xt_xt, psiwkronIndotj )      
#
#
#Gamma_xtm1_xtp1 = 2*gxtm1xtp1
#Gamma_xtm1_xt = 2*gxtm1xt + np.dot(Gamma_xt_xt, psixkronIn) +  np.dot(Gamma_xtm1_xtp1, Inkronpsix)
#
#Gamma_wtp1_wt = np.dot(Gamma_xtp1_wt, psiwkronIk)
#
#
#A_for_psi_x_w = 2*Gamma_xt_2 
#b_for_psi_x_w = - 2*gxtm1wt + np.dot(Gamma_xt_wt, psixkronIk) + np.dot(Gamma_xtm1_xt, Inkronpsiw)
#psi_x_w = np.dot(scipy.linalg.inv(A_for_psi_x_w), b_for_psi_x_w )
#psi_x_w_bysolve = scipy.linalg.solve(A_for_psi_x_w, b_for_psi_x_w)
#
#
#Gamma_xt_wtp1 = 2*gxtwtp1 + 2* np.dot(Gamma_xtp1_2, psi_x_w)  + np.dot(Gamma_xtp1_wtp1, psixkronIk)  + np.dot(Gamma_xt_xtp1, Inkronpsiw)
#Gamma_xtm1_wtp1 = 2*gxtm1wtp1 + np.dot(Gamma_xt_wtp1, psixkronIk) + np.dot(Gamma_xtm1_xtp1, Inkronpsiw)
#
#Gamma_wt_wtp1 = 2*gwtwtp1 + np.dot(Gamma_xt_wtp1, psiwkronIk)
#
##### equation for psi_x_w
## Gammax2*psi_xw + gww +  Gammaxw*(psi_w kron Ik)
#A_for_psi_w_w = Gamma_xt_2 
#b_for_psi_w_w = gwtwt + np.dot(Gamma_xt_wt, psiwkronIk) 
#psi_w_w = np.dot(scipy.linalg.inv(A_for_psi_w_w), b_for_psi_w_w )
#psi_w_w_bysolve = scipy.linalg.solve(A_for_psi_w_w, b_for_psi_w_w)
#
#
#
##Compute Vxx
#utility_sym_mat = sympy.Matrix([fi.utility]) 
#du_dx =  utility_sym_mat.jacobian([fi.xvar_t_sym])   
#du_dx_nopar =  [u.subs(fi.par_to_values_dict) for u in list(du_dx)]
#du_dx_at_ss = [u.subs(fi.normal_and_0_to_ss).subs(fi.ss_solutions_dict) for u in du_dx_nopar]
#du_dx_at_ss = fi.matrix2numpyfloat(sympy.Matrix(du_dx_at_ss))
#
#beta = fi.beta.subs(fi.par_to_values_dict)
#Ibetaphi = np.eye(nx) - beta*psi_x
#invIbetaphi = scipy.linalg.inv(Ibetaphi)
#Vx = np.dot(du_dx_at_ss.T, invIbetaphi)
#        
#du_dx =  utility_sym_mat.jacobian([fi.xvar_t_sym])  
#du_dx_dx = du_dx.jacobian(fi.xvar_t_sym)
#du_dx_dx_nopar =  du_dx_dx.subs(fi.par_to_values_dict)
#du_dx_dx_at_ss = du_dx_dx_nopar.subs(fi.normal_and_0_to_ss).subs(fi.ss_solutions_dict)
#du_dx_dx_at_ss = fi.matrix2numpyfloat(du_dx_dx_at_ss)
##now, vectorize and then transpose the hessian. Becomes 1 \times nx^2 vector
#du_dx_dx_at_ss = du_dx_dx_at_ss.T.reshape(-1,1).T
#
#
#Vxx_term_1 = du_dx_dx_at_ss + beta* np.dot(Vx, psi_x_x)
#inv_of_Vxx_term_2 = np.eye(nx*nx) - beta*psixkronpsix
#Vxx = np.dot(Vxx_term_1, scipy.linalg.inv(inv_of_Vxx_term_2) ) 
#
## Gx
#Gx_term_1 = np.dot(gxtp1, psi_w) + gwtp1
#Gx_term_2 = 2*np.dot(Vx, psi_x_w) + np.dot(Vxx, psixkronpsiw)
#Gx_term_2 = fipir_aug.matrix2numpyfloat(Gx_term_2) 
##matGx_term_2 = Gx_term_2.reshape(nw,nx)
#matGx_term_2 = Gx_term_2.reshape(nw,nx, order='F')
#Gx_term_3 = np.dot(Vxx, psiwkronpsix)
#Gx_term_3 = fipir_aug.matrix2numpyfloat(Gx_term_3) 
##matGx_term_3 = Gx_term_3.reshape(nx,nw)
#matGx_term_3 = Gx_term_3.reshape(nx,nw, order='F')
#
#theta = fi.theta
##foo = matGx_term_2 + matGx_term_3.T
#Gx =  np.dot(Gx_term_1, matGx_term_2 + matGx_term_3.T)/theta
#
#
##Equation (50) bearing on determintation of psi_x_q
## coe1 psixq + coe2 psixq coe3 + constant terms = 0
## convert it as
## (I kron coe1) vec(psixq) + (coe3' kron coe2) vec(psixq) = -vec(constant terms)
## [(I kron coe1) + (coe3' kron coe2)] vec(psixq)  = -vec(constant terms)
## Solve it as Ax = b
#
#
#coe1 = 2*Gamma_xt_2 #is 7x7
#coe2 = 2*Gamma_xtp1_2 # is 7x7
#coe3 = psi_x # is 7x7
#
#
##now some constant terms:
#eq50cons_1 = 2*gxtm1q + np.dot(Gamma_xtm1_xtp1, Inkronpsiq) +  np.dot(Gamma_xtm1_xt, Inkronpsiq)
#
#eq50cons_2_a  = 2*gxtq + 2*np.dot(Gamma_xtp1_q,psi_x) 
#eq50cons_2_b  = np.dot(Gamma_xt_xtp1, Inkronpsiq) + np.dot(Gamma_xt_xt, psiqkronIn) 
#eq50cons_2 = np.dot(eq50cons_2_a+eq50cons_2_b , psi_x)
#
#E_w_dist = - np.dot(Vx,psi_w).T/theta
#eq50cons_3_list = [np.dot(E_w_dist.T, Gamma_xtm1_wtp1[i,:].reshape((nw,nx), order='F')) for i in range(nx)]
#
#eq50cons_3 = np.concatenate(eq50cons_3_list)
#
#eq50cons = eq50cons_1 + eq50cons_2 + eq50cons_3
#
#cons_for_psixq  = eq50cons  + np.dot(Gx, psi_x)
## (coe3' kron coe2) is 49x49, then the I in (I kron coe1) must be 7x7
#A_for_vecpsixq = np.kron(np.eye(nx), coe1) + np.kron(coe3.T, coe2) 
#b_for_vecpsixq = -cons_for_psixq.reshape((-1,1))
#vec_psixq = scipy.linalg.solve(A_for_vecpsixq, b_for_vecpsixq)
#psi_x_q = vec_psixq.reshape((nx,nx), order='F')
#
#### equation for psi_w_q:
## A psiwq + constants = 0
## A is 2*Gamma_{x2}
## psiwq is a matrix and so it is constants. We could vectorize the equation as
## (I kron A) vec(psiwq) = - vec(constants)
#
#eq51_coe = 2* Gamma_xt_2
#eq51_cons_1 = 2*gwtq + np.dot(Gamma_xtp1_wt, psiqkronIk) + np.dot(Gamma_xt_wt, psiqkronIk)
#
#Gamma_xt_q_1of2 = 2*gxtq + 2* np.dot(Gamma_xtp1_2, psi_x_q) + np.dot(Gamma_xtp1_q, psi_x)
#Gamma_xt_q_2of2 = np.dot(Gamma_xt_xtp1, Inkronpsiq)+np.dot(Gamma_xt_xt, psiqkronIn)
#Gamma_xt_q = Gamma_xt_q_1of2 + Gamma_xt_q_2of2
#
#eq51_cons_2 = np.dot(Gamma_xt_q, psi_w)
#
#
#eq51_cons_3a_list = [np.dot(E_w_dist.T, Gamma_wtp1_wt[i,:].reshape((nw,nw), order='F')) for i in range(nx)]
#eq51_cons_3a =  np.concatenate(eq51_cons_3a_list)
#
#eq51_cons_3b_list = [np.dot(E_w_dist.T, Gamma_wt_wtp1[i,:].reshape((nw,nw), order='F')) for i in range(nx)]
#eq51_cons_3b =  np.concatenate(eq51_cons_3b_list)
#
#eq51_cons_3 = eq51_cons_3a + eq51_cons_3b
#
#eq51_cons = eq51_cons_1 + eq51_cons_2 + eq51_cons_3
#cons_for_psiwq  = eq51_cons  + np.dot(Gx, psi_w)
#
#A_for_vecpsiwq = np.kron(np.eye(nw,nw), eq51_coe)
#b_for_vecpsiwq = -cons_for_psiwq.reshape((-1,1))
#vec_psiwq = scipy.linalg.solve(A_for_vecpsiwq, b_for_vecpsiwq)
#psi_w_q = vec_psiwq.reshape((nx,nw), order='F')
# 
# 
#### equation for Vxq:
##du_dx =  utility_sym_mat.jacobian([fi.xvar_t_sym])  
#du_dx_dq = du_dx.jacobian(sympy.Matrix([q]))
#du_dx_dq_nopar =  du_dx_dq.subs(fi.par_to_values_dict)
#du_dx_dq_at_ss = du_dx_dq_nopar.subs(fi.normal_and_0_to_ss).subs(fi.ss_solutions_dict)
#du_dx_dq_at_ss = fi.matrix2numpyfloat(du_dx_dq_at_ss)
##now, vectorize and then transpose the hessian. Becomes 1 \times nx^2 vector
#du_dx_dq_at_ss = du_dx_dq_at_ss.T.reshape(-1,1).T
#uxq = du_dx_dq_at_ss
#
#Vxiq_term1 = beta*np.dot(Vx, psi_x_q)+0.5*np.dot(Vxx,psixkronpsiq+psiqkronpsix)
#
#Vxq_term2a = 2*np.dot(Vx, psi_x_w) + np.dot(Vxx, psixkronpsiw)
#Vxq_term2a =   Vxq_term2a.reshape((nw, nx), order='F')
#Vxq_term2b =  np.dot(Vxx, psiwkronpsix)
#Vxq_term2b =   Vxq_term2b.reshape((nx, nw), order='F')
#Vxq_term2c =   np.dot(Vx,psi_w)
#Vxq_term2 = -beta*0.5*np.dot(Vxq_term2c,(Vxq_term2a + Vxq_term2b.T))/theta
#Vxq_rhs = uxq + Vxiq_term1 + Vxq_term2
#Vxq_lhs = np.eye(nx) - beta*psi_x
##Vxq = np.dot(Vxq_rhs, scipy.linalg.inv(Vxq_lhs))
#Vxq = scipy.linalg.solve(Vxq_lhs.T, Vxq_rhs.T).T #solving A.T x.T = b.T
#
#
#### equation for psi_q_q
#Eq52_coe = Gamma_xtp1_2 + Gamma_xt_2
#Eq52_cons_1 = gqq + np.dot(Gamma_xtp1_q + Gamma_xt_q, psi_q)
#
#Gamma_wtp1_q_a = 2*gwtp1q + 2*np.dot(Gamma_xtp1_2, psi_w_q) + np.dot(Gamma_xtp1_wtp1, psiqkronIk)
#Gamma_wtp1_q_b = np.dot(Gamma_xtp1_q, psi_w) + np.dot(Gamma_xt_wtp1, psiqkronIk) 
#Gamma_wtp1_q = Gamma_wtp1_q_a + Gamma_wtp1_q_b 
#
#Eq52_cons_2a = np.dot(Gamma_wtp1_q, E_w_dist)
#
#Gamma_wtp1_wtp1 = gwtp1wtp1 + np.dot(Gamma_xtp1_2, psi_w_w)+np.dot(Gamma_xtp1_wtp1, psiwkronIk) 
#
#E_wkronw_dist = np.eye(nw) + np.dot( np.dot(Vx,psi_w).T, np.dot(Vx,psi_w))/(theta**2)
#E_wkronw_dist = E_wkronw_dist.reshape((-1,1))
#Eq52_cons_2b = np.dot(Gamma_wtp1_wtp1, E_wkronw_dist)
#Eq52_cons_2 = Eq52_cons_2a + Eq52_cons_2b 
#Eq52_cons = Eq52_cons_1 + Eq52_cons_2 
#
#
#cons_psiqq_a = (np.dot(gxtp1, psi_w) + gwtp1)/theta
#
#cons_psiqq_b =   2*np.dot(Vx, psi_w_q) + np.dot(Vxx, (psiwkronpsiq+psiqkronpsiw)) +  2*np.dot(Vxq, psi_w)
#cons_psiqq_ab  = np.dot(cons_psiqq_a , cons_psiqq_b.T)
# 
#cons_psiqq_c_1 = (np.dot(Vx, psi_w_w) + np.dot(Vxx, psiwkronpsiw)).reshape((nw,nw), order='F')
#cons_psiqq_c = np.dot(cons_psiqq_c_1.T, E_w_dist)
#cons_psiqq_ac  = np.dot(cons_psiqq_a , cons_psiqq_c)
#
#cons_psiqq_d = np.dot(cons_psiqq_c_1, E_w_dist)
#cons_psiqq_ad  = np.dot(cons_psiqq_a , cons_psiqq_d)
#
#terms_in_Eq56 =  np.dot(Gx, psi_q) - cons_psiqq_ab - cons_psiqq_ac - cons_psiqq_ad
##terms_in_Eq56 =  np.dot(Gx, psi_q) + cons_psiqq_ab - cons_psiqq_ac - cons_psiqq_ad
#
#cons_for_psiqq = Eq52_cons + terms_in_Eq56 
#
#psi_q_q = scipy.linalg.solve(Eq52_coe, -cons_for_psiqq )
#
#
#print "\nEnd of script!"
#
#
