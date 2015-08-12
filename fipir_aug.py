# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 11:12:37 2014

@author: ricardomayerb
"""

import sympy
from scipy import linalg, optimize
import numpy as np
import matplotlib.pyplot as plt

from sympy.printing.theanocode import theano_function


# if using R's geigen package to compute the ordered QZ decomposition, we
# need to load this package via rpy2:
# Note: Scipy 0.17 plans to pack its own ordered QZ
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
robjects.r.library("geigen")
rgqz = robjects.r.gqz


q = sympy.Symbol('q')

#x_names_list = ['a', 'C','I', 'K', 'L','Y','zetaa']
#w_names_list = ['wa', 'wzetaa']

x_names = []
w_names = []
x_sym_dict = {}
xss_sym_dict = {}
w_sym_dict = {}
wss_sym_dict = {}

#param_sym_dict = {}


def get_sstate_sol_dict_from_sympy_eqs(glist, xss_sym_dict, xini_dict={}):

#    print xss_sym_dict
#    print xini_dict
#    print glist
    
    nx = len(glist)

    glist_lam = sympy.lambdify([xss_sym_dict.values()], glist, dummify=False)

    glist_lam

    if xini_dict == {}:
        xini_list = 0.2*np.ones(nx)
#        print '\nxini_list:', xini_list
    else:
        xini_list = [x.subs(xini_dict) for x in xss_sym_dict.values()]

#    print xini_list
    
    xini_af = np.empty(len(xini_list), dtype='float')

    for i in range(len(xini_list)):
        xini_af[i] = xini_list[i]

    xini_list = np.array(xini_list)

    sol = optimize.root(glist_lam, xini_af)

    return dict(zip(xss_sym_dict.values(), np.around(sol['x'], decimals=12)))


def make_wss_to_zero_dict(w_names):

    wss_zero_dict = {}
    ss_names = [name + '_ss' for name in w_names]
    ss_names_sym = [sympy.Symbol(x) for x in ss_names]
    zero_list = [0 for i in w_names]
    wss_zero_dict.update(dict(zip(ss_names_sym, zero_list)))

    return wss_zero_dict


def make_param_sym_dict(param_names):
    dic_to_be_filled = {}
    names_str = param_names
    names_sym = [sympy.Symbol(x) for x in names_str]
    dic_to_be_filled.update(dict(zip(names_str, names_sym)))

    return dic_to_be_filled


def make_sym_dict(names_list, date_str, isw=False):
    dic_to_be_filled = {}
    dated_names = [name + date_str for name in names_list]
    dated_names_sym = [sympy.Symbol(x) for x in dated_names]
    dic_to_be_filled.update(dict(zip(dated_names, dated_names_sym)))

    if isw:
        return dic_to_be_filled

    dated_names_0 = [name + '_0' for name in dated_names]
    dated_names_0_sym = [sympy.Symbol(x) for x in dated_names_0]
    dic_to_be_filled.update(dict(zip(dated_names_0, dated_names_0_sym)))

    dated_names_1 = [name + '_1' for name in dated_names]
    dated_names_1_sym = [sympy.Symbol(x) for x in dated_names_1]
    dic_to_be_filled.update(dict(zip(dated_names_1, dated_names_1_sym)))

    dated_names_2 = [name + '_2' for name in dated_names]
    dated_names_2_sym = [sympy.Symbol(x) for x in dated_names_2]
    dic_to_be_filled.update(dict(zip(dated_names_2, dated_names_2_sym)))

    dated_names_q = [name + '_q' for name in dated_names]
    dated_names_q_fun = [sympy.Function(x)(q) for x in dated_names]
#    print '\n'
    dic_to_be_filled.update(dict(zip(dated_names_q, dated_names_q_fun)))

    return dic_to_be_filled


def make_qdiff_to_q012(x_names):
    qdiff_to_q012 = {}
    qdiff_0 = {}
    qdiff_1 = {}
    qdiff_2 = {}

    for date_str in ['tp1', 't', 'tm1']:
        dated_names = [name + date_str for name in x_names]
        dated_q_fun_sym = [sympy.Function(x)(q) for x in dated_names]

        dated_names_0 = [name + '_0' for name in dated_names]
        dated_names_0_sym = [sympy.Symbol(x) for x in dated_names_0]
        qdiff_0.update(dict(zip(dated_q_fun_sym, dated_names_0_sym)))

        dated_qdiffs_1 = [sympy.diff(x, q, 1) for x in dated_q_fun_sym]
        dated_names_1 = [name + '_1' for name in dated_names]
        dated_names_1_sym = [sympy.Symbol(x) for x in dated_names_1]
        qdiff_1.update(dict(zip(dated_qdiffs_1, dated_names_1_sym)))

        dated_qdiffs_2 = [sympy.diff(x, q, 2) for x in dated_q_fun_sym]
        dated_names_2 = [name + '_2' for name in dated_names]
        dated_names_2_sym = [sympy.Symbol(x) for x in dated_names_2]
        qdiff_2.update(dict(zip(dated_qdiffs_2, dated_names_2_sym)))

    qdiff_to_q012.update(qdiff_0)
    qdiff_to_q012.update(qdiff_1)
    qdiff_to_q012.update(qdiff_2)

    return qdiff_to_q012


def make_normal_to_steady_state(x_names, w_names):

    xw_to_ss_d = {}

    x_to_ss_d = {}
    x_ss_names = [name + '_ss' for name in x_names]
    x_ss_sym = [sympy.Symbol(x) for x in x_ss_names]

    w_to_ss_d = {}
    w_ss_names = [name + '_ss' for name in w_names]
    w_ss_sym = [sympy.Symbol(x) for x in w_ss_names]

    for date_str in ['tm1', 't', 'tp1']:
        dated_names = [name + date_str for name in x_names]
        dated_names_0 = [name + '_0' for name in dated_names]
        x_to_ss_d.update(dict(zip(dated_names, x_ss_sym)))
        x_to_ss_d.update(dict(zip(dated_names_0, x_ss_sym)))

    for date_str in ['tp1', 't']:
        dated_names = [name + date_str for name in w_names]
        w_to_ss_d.update(dict(zip(dated_names, w_ss_sym)))

    xw_to_ss_d.update(x_to_ss_d)
    xw_to_ss_d.update(w_to_ss_d)

    return xw_to_ss_d


def make_normal_to_q_dict(x_names, w_names):

    x_normal_to_q = {}
    w_normal_to_q = {}
    xw_normal_to_q = {}

    for date_str in ['tp1', 't', 'tm1']:
        dated_names = [name + date_str for name in x_names]
        dated_q_fun_sym = [sympy.Function(x)(q) for x in dated_names]
        x_normal_to_q.update(dict(zip(dated_names, dated_q_fun_sym)))

    for date_str in ['tp1', 't']:
        dated_names = [name + date_str for name in w_names]
        dated_q_mul_sym = [q*sympy.Symbol(x) for x in dated_names]
        w_normal_to_q.update(dict(zip(dated_names, dated_q_mul_sym)))

    xw_normal_to_q.update(x_normal_to_q)
    xw_normal_to_q.update(w_normal_to_q)

    return xw_normal_to_q


def set_param_sym_dict(this_param_names):

    param_sym_dict = {}
    names_sym = [sympy.Symbol(x) for x in this_param_names]

    this_param_sym_dict = dict(zip(this_param_names, names_sym))
    param_sym_dict.update(this_param_sym_dict)

    return param_sym_dict


def make_sym_dict_all_dates(names_list, dates_list, isw=True, do_steady_states_names=True):

    sym_dict_all_dates = {}

    for date in dates_list:
        this_dict = make_sym_dict(names_list, date, isw=isw)
        sym_dict_all_dates.update(this_dict)

    if do_steady_states_names:
        names_steady_state_list = [name + '_ss' for name in names_list]
        names_steady_state_list_sym = [
            sympy.Symbol(x) for x in names_steady_state_list]
        steady_state_sym_dict = dict(
            zip(names_steady_state_list, names_steady_state_list_sym))
        sym_dict_all_dates.update(steady_state_sym_dict)
        return sym_dict_all_dates, steady_state_sym_dict

    else:
        return sym_dict_all_dates


def set_x_w_sym_dicts(this_x_names, this_w_names):

    x_names = this_x_names
    w_names = this_w_names

#    print '\nx_names, this_x_names: ',x_names, this_x_names

    x_dates = ['tm1', 't', 'tp1']
    w_dates = ['t', 'tp1']

    x_sym, x_sym_ss = make_sym_dict_all_dates(x_names, x_dates)
    w_sym, w_sym_ss = make_sym_dict_all_dates(w_names, w_dates, isw=True)

    x_sym_dict = {}

    x_sym_dict.update(x_sym)
    x_sym_dict.update({'q': q})
    xss_sym_dict.update(x_sym_ss)
    w_sym_dict.update(w_sym)
    wss_sym_dict.update(w_sym_ss)

    x_sym_tm1_dict = make_sym_dict(x_names, 'tm1')
    x_sym_t_dict = make_sym_dict(x_names, 't')
    x_sym_tp1_dict = make_sym_dict(x_names, 'tp1')
    w_sym_t_dict = make_sym_dict(w_names, 't')
    w_sym_tp1_dict = make_sym_dict(w_names, 'tp1')
#
    return x_sym_dict, w_sym_dict

def make_x_w_param_sym_dicts(this_x_names, this_w_names, this_param_names):
    x_sym_dict, w_sym_dict = set_x_w_sym_dicts(this_x_names, this_w_names)
    param_sym_dict = set_param_sym_dict(this_param_names)
    
    x_in_ss_sym_d = {st: x_sym_dict[st] for st in x_sym_dict.keys() 
                     if 'ss' in st}

    
    return x_sym_dict, x_in_ss_sym_d, w_sym_dict, param_sym_dict




def Fckliy_az_mua(xini_az):
    barC, barK, barL, barI, barY, bara, barzetaa = xini_az[0], xini_az[
        1], xini_az[2], xini_az[3], xini_az[4], xini_az[5], xini_az[6]

    foc1 = beta*np.exp(-mua)*(alpha*barY/barK + 1 - delta) - 1
    foc2 = barY - barC - barI
    foc3 = barI - np.exp(mua)*barK + (1-delta)*barK
    nurat = nu/(1-nu)
    foc4 = nurat*(1-alpha)*barY/barC - barL/(1-barL)
    foc5 = barY - np.exp((1-alpha)*mua) * barK**alpha * barL**(1-alpha)
    foc6 = bara - mua - barzetaa
    foc7 = barzetaa - rhoavalue*barzetaa
    focsckliy = np.array([foc1, foc2, foc3, foc4, foc5, foc6, foc7])

    return focsckliy


def make_state_space_sym(num_signals, num_states, is_homo):

    mu_rho_dict_sym = {}
    state_state_sym_dict = {}

    mu_names = ['mu_'+str(i) for i in range(num_states)]
    rho_names = ['rho_'+str(i) for i in range(num_states)]
    mu_rho_names = mu_names + rho_names

    mu_names_sym = [sympy.Symbol(x) for x in mu_names]
    rho_names_sym = [sympy.Symbol(x) for x in rho_names]
    mu_rho_names_sym = [sympy.Symbol(x) for x in mu_rho_names]
    mu_rho_dict_sym.update(dict(zip(mu_rho_names, mu_rho_names_sym)))

    A_identity_sym = sympy.eye(num_states)
    A_rhoblock_sym = sympy.diag(*rho_names_sym)
    A_sym = sympy.diag(A_identity_sym, A_rhoblock_sym)

    D_sym_mu_part = sympy.ones(num_signals, num_states)
    D_sym_zeta_part = sympy.ones(num_signals, num_states)
    D_sym = D_sym_mu_part.row_join(D_sym_zeta_part)

    if is_homo:
        sigmas_signal_names = [
            'sigma_signal_'+str(i) for i in range(num_signals)]
        sigmas_state_names = ['sigma_state_'+str(i) for i in range(num_states)]
        sigmas_signal_sym = [sympy.Symbol(x) for x in sigmas_signal_names]
        sigmas_state_sym = [sympy.Symbol(x) for x in sigmas_state_names]

        C_nonsingularblock_sym = sympy.diag(*sigmas_state_sym)
        G_nonsingularblock_sym = sympy.diag(*sigmas_signal_sym)

        C_singularblock_sym = sympy.zeros(num_states, num_states)
        G_singularblock_sym = sympy.zeros(num_signals, num_states)

        G_sym = G_singularblock_sym.row_join(G_nonsingularblock_sym)
        C_sym = sympy.diag(C_singularblock_sym, C_nonsingularblock_sym)

    main_matrices_sym = {
        'A_z': A_sym, 'C_z': C_sym, 'D_s': D_sym, 'G_s': G_sym}
    sub_matrices_sym = {'A_z_stable': A_rhoblock_sym,
                        'C_z_nonsingular': C_nonsingularblock_sym,
                        'G_s_nonsingular': G_nonsingularblock_sym}

    state_state_sym_dict.update(main_matrices_sym)
    state_state_sym_dict.update(sub_matrices_sym)

    return state_state_sym_dict


def all_vars_par_par(a_of_values, set_par_x, set_par_lev, par_tex_x,
                     par_tex_lev, par_name_x, par_name_lev, vars_tex, postname):

    for v in range(len(set_par_lev)):
        fig, ax = plt.subplots()
        for i in range(5):
            ax.plot(set_par_x, a_of_values[i, v, :], label=vars_tex[i])

        ax.legend(loc="best", ncol=2)
        ax.grid(True)
        ax.set_title("Coeff of persistent shock " +
                     "("+par_tex_lev + " = " +
                     "{0:.2f})".format(set_par_lev[v]))
        ax.set_xlabel(par_tex_x)
        filename = "../../figures/psi_change_" + par_name_x + "_" + \
            par_name_lev + "_all_vars_" + str(v) + "_" + postname + ".pdf"
        fig.savefig(filename)


def ir_toy_rpi_from(azA_mat, x_sta, x_nsta, BGP_det, phi_q, islog=False):

    copy_phi_q = phi_q.copy()
    copy_phi_q.shape = (len(phi_q), 1)

    x_sta_plus_phiq = x_sta + copy_phi_q

    return x_sta_plus_phiq, azA_mat, x_sta, x_nsta, BGP_det


def matrix2numpyfloat(m):
    a = np.empty(m.shape, dtype='float')
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            a[i, j] = m[i, j]
    return a


def old_matrix2numpyfloat(m):
    a = np.empty(m.shape, dtype='float')
    for i in range(m.rows):
        for j in range(m.cols):
            a[i, j] = m[i, j]
    return a


def numpy_state_space_matrices(sspacemats_dic, name2val_dic, user_names=False,
                               mod2user_dic=None):
                                   
    A = sspacemats_dic['A_z']
    C = sspacemats_dic['C_z']
    D = sspacemats_dic['D_s']
    G = sspacemats_dic['G_s']
                                  

    if mod2user_dic:
        Asn = A.subs(mod2user_dic).subs(name2val_dic)
        Csn = C.subs(mod2user_dic).subs(name2val_dic)
        Dsn = D.subs(mod2user_dic).subs(name2val_dic)
        Gsn = G.subs(mod2user_dic).subs(name2val_dic)
    else:
        Asn = A.subs(name2val_dic)
        Csn = C.subs(name2val_dic)
        Dsn = D.subs(name2val_dic)
        Gsn = G.subs(name2val_dic)

    Ann = matrix2numpyfloat(Asn)
    Cnn = matrix2numpyfloat(Csn)
    Dnn = matrix2numpyfloat(Dsn)
    Gnn = matrix2numpyfloat(Gsn)

    return Ann, Cnn, Dnn, Gnn

def numpy_state_space_matrices_old(A, C, D, G, name2val_dic, user_names=False,
                               mod2user_dic=None):

    if mod2user_dic:
        Asn = A.subs(mod2user_dic).subs(name2val_dic)
        Csn = C.subs(mod2user_dic).subs(name2val_dic)
        Dsn = D.subs(mod2user_dic).subs(name2val_dic)
        Gsn = G.subs(mod2user_dic).subs(name2val_dic)
    else:
        Asn = A.subs(name2val_dic)
        Csn = C.subs(name2val_dic)
        Dsn = D.subs(name2val_dic)
        Gsn = G.subs(name2val_dic)

    Ann = matrix2numpyfloat(Asn)
    Cnn = matrix2numpyfloat(Csn)
    Dnn = matrix2numpyfloat(Dsn)
    Gnn = matrix2numpyfloat(Gsn)

    return Ann, Cnn, Dnn, Gnn

class SignalStateSpace:

    def __init__(self, n_x=1, n_s=1, param_val_dic={},
                 user_names_dic={}):
                     
        ssp_mats_sym_d = make_state_space_sym(n_s,
                                              n_x, True)
        
        self.A_sym = ssp_mats_sym_d['A_z']
        self.C_sym = ssp_mats_sym_d['C_z']
        self.D_sym = ssp_mats_sym_d['D_s']
        self.G_sym = ssp_mats_sym_d['G_s']
        
        if  user_names_dic!={}:  
            self.A_sym_u = self.A_sym.subs(user_names_dic)
            self.C_sym_u = self.C_sym.subs(user_names_dic)
            self.D_sym_u = self.D_sym.subs(user_names_dic)
            self.G_sym_u = self.G_sym.subs(user_names_dic)
            self.A_num = self.A_sym_u.subs(param_val_dic) 
            self.A_num = matrix2numpyfloat(self.A_num) 
            self.C_num = self.C_sym_u.subs(param_val_dic) 
            self.C_num = matrix2numpyfloat(self.C_num) 
            self.D_num = self.D_sym_u.subs(param_val_dic) 
            self.D_num = matrix2numpyfloat(self.D_num) 
            self.G_num = self.G_sym_u.subs(param_val_dic) 
            self.G_num = matrix2numpyfloat(self.G_num) 
                        
        else:
            self.A_num = self.A_sym.subs(param_val_dic) 
            self.A_num = matrix2numpyfloat(self.A_num) 
            self.C_num = self.C_sym.subs(param_val_dic) 
            self.C_num = matrix2numpyfloat(self.C_num) 
            self.D_num = self.D_sym.subs(param_val_dic) 
            self.D_num = matrix2numpyfloat(self.D_num) 
            self.G_num = self.G_sym.subs(param_val_dic) 
            self.G_num = matrix2numpyfloat(self.G_num) 
            self.A_sym_u = {}
            self.C_sym_u = {}
            self.D_sym_u = {}
            self.G_sym_u = {}
            
           
            


class OldSignalStateSpace:

    def __init__(self, A_num=[], C_num=[], D_num=[], G_num=[],
                 A_sym=None, C_sym=None, D_sym=None, G_sym=None, param_val_dic={},
                 user_names_dic={}):

        self.A_num = A_num
        self.C_num = C_num
        self.D_num = D_num
        self.G_num = G_num

        self.A_sym = A_sym
        self.C_sym = C_sym
        self.D_sym = D_sym
        self.G_sym = G_sym

        if param_val_dic != {}:
            self.A_num = A_sym.subs(param_val_dic)
            self.A_num = matrix2numpyfloat(self.A_num)
            self.C_num = C_sym.subs(param_val_dic)
            self.C_num = matrix2numpyfloat(self.C_num)
            self.D_num = D_sym.subs(param_val_dic)
            self.D_num = matrix2numpyfloat(self.D_num)
            self.G_num = G_sym.subs(param_val_dic)
            self.G_num = matrix2numpyfloat(self.G_num)

        if user_names_dic != {}:
            self.A_sym = self.A_sym.subs(user_names_dic)
            self.C_sym = self.C_sym.subs(user_names_dic)
            self.D_sym = self.D_sym.subs(user_names_dic)
            self.G_sym = self.G_sym.subs(user_names_dic)


class FullInfoModel:
    x = 0

    def __init__(self, statespace, all_names,
                 xw_sym_dicts={}, ss_x_sol_dict={},  par_to_values_dict={},
                 eq_conditions=[], utility=[], xss_ini_dict={}, theta=10.0):

        self.statespace = statespace
        self.A_num = statespace.A_num
        self.C_num = statespace.C_num
        self.D_num = statespace.D_num
        self.G_num = statespace.G_num
        self.par_to_values_dict = par_to_values_dict
        self.x_names = all_names['x_names']
        self.w_names = all_names['w_names']
        self.param_names = all_names['param_names']
        self.q = sympy.Symbol('q')
        self.utility = utility
        self.beta = sympy.Symbol('beta')
        self.theta = theta

        self.param_sym_dict = make_param_sym_dict(self.param_names)

        if xw_sym_dicts == {}:
            self.xw_sym_dicts = set_x_w_sym_dicts(self.x_names, self.w_names)
        else:
            self.xw_sym_dicts = xw_sym_dicts

        self.x_s_d, self.w_s_d = self.xw_sym_dicts

        self.normal_x_s_d = {st: self.x_s_d[st] for st in self.x_s_d.keys() if
                             not ('_q' in st or '_1' in st or '_2' in st or '_0' in st
                                  or 'ss' in st or 'q' in st)}

        self.normal_x_s_tp1 = {st: self.normal_x_s_d[st] for st in self.normal_x_s_d.keys() if
                               'tp1' in st}

        self.normal_x_s_t = {st: self.normal_x_s_d[st] for st in self.normal_x_s_d.keys() if
                             not('tm1' in st or 'tp1' in st)}

        self.normal_x_s_tm1 = {st: self.normal_x_s_d[st] for st in self.normal_x_s_d.keys() if
                               'tm1' in st}

        self.normal_w_s_d = {st: self.w_s_d[st] for st in self.w_s_d.keys() if
                             not 'ss' in st}

        self.normal_w_s_tp1 = {st: self.normal_w_s_d[st] for st in self.normal_w_s_d.keys() if
                               'tp1' in st}

        self.normal_w_s_t = {st: self.normal_w_s_d[st] for st in self.normal_w_s_d.keys() if
                             not('tm1' in st or 'tp1' in st)}

        self.xvar_tp1_sym = self.normal_x_s_tp1.values()
        self.xvar_t_sym = self.normal_x_s_t.values()
        self.xvar_tm1_sym = self.normal_x_s_tm1.values()
        self.wvar_tp1_sym = self.normal_w_s_tp1.values()
        self.wvar_t_sym = self.normal_w_s_t.values()

        self.normal_xw_to_q = make_normal_to_q_dict(self.x_names, self.w_names)

        self.normal_and_0_to_ss = make_normal_to_steady_state(self.x_names, self.w_names)

        self.x_in_ss_sym_d = {st: self.x_s_d[st]
                              for st in self.x_s_d.keys() if 'ss' in st}
        
#        print 'self.x_in_ss_sym_d', self.x_in_ss_sym_d
        
        self.w_in_ss_zero_d = make_wss_to_zero_dict(self.w_names)

        self.qdiffs_to_012_d = make_qdiff_to_q012(self.x_names)

        self.eq_conditions = eq_conditions
        self.d_g_dxw_1, self.d_g_dxw_2 = self.d1d2_g_x_w_unevaluated()

        self.fun_d_first_numpy = None
        self.fun_d_second_numpy = None
        self.Need_compile_theano_fn_first = False
        self.Need_compile_theano_fn_second = False
#        self.fun_d_first_theano = None
#        self.fun_d_second_theano = None


#        symbols_per_eq = [g.atoms(sympy.Symbol) for g in self.eq_conditions]
#        self.xwparams_symbols_eq_cond = list(set.union(*symbols_per_eq))

        if par_to_values_dict != {}:
            self.eq_conditions_nopar = [
                x.subs(par_to_values_dict) for x in self.eq_conditions]
            self.eq_conditions_nopar_ss = [x.subs(self.normal_and_0_to_ss).subs(
                self.w_in_ss_zero_d) for x in self.eq_conditions_nopar]

            self.eq_conditions_matrix = sympy.Matrix(self.eq_conditions)
            symbols_per_eq_nopar = [
                g.atoms(sympy.Symbol) for g in self.eq_conditions_nopar]
            self.xw_symbols_eq_cond_nopar = list(
                set.union(*symbols_per_eq_nopar))

#            print 'self.eq_conditions_nopar', self.eq_conditions_nopar
#            print 'self.normal_and_0_to_ss', self.normal_and_0_to_ss
#            print 'self.w_in_ss_zero_d', self.w_in_ss_zero_d
#            print 'self.eq_conditions_nopar_ss',self.eq_conditions_nopar_ss
            
            if ss_x_sol_dict == {}:
                self.ss_solutions_dict = get_sstate_sol_dict_from_sympy_eqs(
                    self.eq_conditions_nopar_ss,
                    self.x_in_ss_sym_d,
                    xini_dict=xss_ini_dict)
            else:
                self.ss_solutions_dict = ss_x_sol_dict

            self.xss_ini_dict = self.ss_solutions_dict

            normal_x_s_tp1_ss_values = [x.subs(self.normal_and_0_to_ss).subs(self.ss_solutions_dict)
                                        for x in self.normal_x_s_tp1.values()]
            self.normal_x_s_tp1_ss_values_d = dict(
                zip(self.normal_x_s_tp1.values(), normal_x_s_tp1_ss_values))

            normal_x_s_t_ss_values = [x.subs(self.normal_and_0_to_ss).subs(self.ss_solutions_dict)
                                      for x in self.normal_x_s_t.values()]
            self.normal_x_s_t_ss_values_d = dict(
                zip(self.normal_x_s_t.values(), normal_x_s_t_ss_values))


            normal_x_s_tm1_ss_values = [x.subs(self.normal_and_0_to_ss).subs(self.ss_solutions_dict)
                                        for x in self.normal_x_s_tm1.values()]
            self.normal_x_s_tm1_ss_values_d = dict(
                zip(self.normal_x_s_tm1.values(), normal_x_s_tm1_ss_values))

            self.normal_x_s_ss_values_d = {}
            self.normal_x_s_ss_values_d.update(self.normal_x_s_tp1_ss_values_d)
            self.normal_x_s_ss_values_d.update(self.normal_x_s_t_ss_values_d)
            self.normal_x_s_ss_values_d.update(self.normal_x_s_tm1_ss_values_d)

            self.normal_w_tp1_s_zero_pairs = [
                (x, 0) for x in self.normal_w_s_d.values()]
            self.normal_w_s_ss_values_d = dict(self.normal_w_tp1_s_zero_pairs)

            self.normal_xw_s_ss_values_d = {}
            self.normal_xw_s_ss_values_d.update(self.normal_x_s_ss_values_d)
            self.normal_xw_s_ss_values_d.update(self.normal_w_s_ss_values_d)

            args_values_x = [x.subs(self.normal_x_s_ss_values_d)
                             for x in self.normal_x_s_d.values()]

            args_values_w = [w.subs(self.normal_w_s_ss_values_d)
                             for w in self.normal_w_s_d.values()]

            args_values_p = [p.subs(par_to_values_dict)
                             for p in self.param_sym_dict.values()]

            self.args_values = args_values_x + args_values_w + args_values_p


#        self.ss_solutions_dict =  get_sstate_sol_dict_from_sympy_eqs(
#                            eqs_nopar_ss, self.x_in_ss_sym_d,
#                                           xini_dict=self.xss_ini_dict)
#


#        for k, v in x_sym_dict.iteritems():
#        for k, v in self.x_s_d.iteritems() :
#            exec('self.' + k + '= v')
#
#
#        for k, v in w_sym_dict.iteritems():
#            exec('self.' + k + '= v')
#
#
#        for k, v in param_sym_dict.iteritems():
#            exec('self.' + k + '= v')

    def d1d2_g_implicit_q_unevaluated(self):

        g_eqs = self.eq_conditions
        g_eqs_q = [x.subs(self.normal_xw_to_q) for x in g_eqs]

        d_g_d_q_1 = [sympy.diff(x, q, 1) for x in g_eqs_q]
        d_g_d_q_2 = [sympy.diff(x, q, 2) for x in g_eqs_q]

        d_g_d_q_1_as012vars = [x.subs(self.qdiffs_to_012_d) for x in d_g_d_q_1]
        d_g_d_q_2_as012vars = [x.subs(self.qdiffs_to_012_d) for x in d_g_d_q_2]

        d_g_d_q_1_012_ss = [x.subs(self.normal_and_0_to_ss)
                            for x in d_g_d_q_1_as012vars]
        d_g_d_q_2_012_ss = [x.subs(self.normal_and_0_to_ss)
                            for x in d_g_d_q_2_as012vars]

        print '\nd_g_d_q_1: ', d_g_d_q_1
        print '\nd_g_d_q_1_as012vars: ', d_g_d_q_1_as012vars
        print '\nd_g_d_q_1_012_ss: ', d_g_d_q_1_012_ss

    def get_evaluated_dgdxw12(self, eq_system=[], param_vals_d={}, mod='numpy'):

        if eq_system == []:
            #            print '\nyes'
            eqs = self.eq_conditions
            d_g_dxw_1, d_g_dxw_2 = self.d_g_dxw_1, self.d_g_dxw_2
            if param_vals_d == {}:
                #                print '\nyes again'
                param_vals_d = self.par_to_values_dict
                ss_sol_dict = self.ss_solutions_dict
                args_values = self.args_values
            else:
                print '\nsame eqs, different par_vals'
                eqs_nopar = [x.subs(param_vals_d) for x in eqs]
                eqs_nopar_ss = [x.subs(self.normal_and_0_to_ss).subs(self.w_in_ss_zero_d)
                                for x in eqs_nopar]
                ss_sol_dict = get_sstate_sol_dict_from_sympy_eqs(
                    eqs_nopar_ss, self.x_in_ss_sym_d,
                    xini_dict=self.xss_ini_dict)

                normal_x_s_tp1_ss_values = [x.subs(self.normal_and_0_to_ss).subs(ss_sol_dict)
                                            for x in self.normal_x_s_tp1.values()]
                normal_x_s_tp1_ss_values_d = dict(
                    zip(self.normal_x_s_tp1.values(), normal_x_s_tp1_ss_values))

                normal_x_s_t_ss_values = [x.subs(self.normal_and_0_to_ss).subs(ss_sol_dict)
                                          for x in self.normal_x_s_t.values()]
                normal_x_s_t_ss_values_d = dict(
                    zip(self.normal_x_s_t.values(), normal_x_s_t_ss_values))

                normal_x_s_tm1_ss_values = [x.subs(self.normal_and_0_to_ss).subs(ss_sol_dict)
                                            for x in self.normal_x_s_tm1.values()]
                normal_x_s_tm1_ss_values_d = dict(
                    zip(self.normal_x_s_tm1.values(), normal_x_s_tm1_ss_values))

                normal_x_s_ss_values_d = {}
                normal_x_s_ss_values_d.update(normal_x_s_tp1_ss_values_d)
                normal_x_s_ss_values_d.update(normal_x_s_t_ss_values_d)
                normal_x_s_ss_values_d.update(normal_x_s_tm1_ss_values_d)

                normal_w_tp1_s_zero_pairs = [(x, 0)
                                             for x in self.normal_w_s_d.values()]
                normal_w_s_ss_values_d = dict(normal_w_tp1_s_zero_pairs)

                args_values_x = [x.subs(normal_x_s_ss_values_d)
                                 for x in self.normal_x_s_d.values()]

                args_values_w = [w.subs(normal_w_s_ss_values_d)
                                 for w in self.normal_w_s_d.values()]

                args_values_p = [p.subs(param_vals_d)
                                 for p in self.param_sym_dict.values()]

                args_values = args_values_x + args_values_w + args_values_p

        else:
            print 'nooooo, different eqs'
            eqs = eq_system
            d_g_dxw_1, d_g_dxw_2 = self.d1d2_g_x_w_unevaluated()
            if param_vals_d == {}:
                param_vals_d = self.par_to_values_dict
            else:
                print '\ndifferent eqs, different par_vals'
                eqs_nopar = [x.subs(param_vals_d) for x in eqs]
                eqs_nopar_ss = [x.subs(self.normal_and_0_to_ss).subs(self.w_in_ss_zero_d)
                                for x in eqs_nopar]
                ss_sol_dict = get_sstate_sol_dict_from_sympy_eqs(
                    eqs_nopar_ss, self.x_in_ss_sym_d,
                    xini_dict=self.xss_ini_dict)

                normal_x_s_tp1_ss_values = [x.subs(self.normal_and_0_to_ss).subs(ss_sol_dict)
                                            for x in self.normal_x_s_tp1.values()]
                normal_x_s_tp1_ss_values_d = dict(
                    zip(self.normal_x_s_tp1.values(), normal_x_s_tp1_ss_values))

                normal_x_s_t_ss_values = [x.subs(self.normal_and_0_to_ss).subs(ss_sol_dict)
                                          for x in self.normal_x_s_t.values()]
                normal_x_s_t_ss_values_d = dict(
                    zip(self.normal_x_s_t.values(), normal_x_s_t_ss_values))

                normal_x_s_tm1_ss_values = [x.subs(self.normal_and_0_to_ss).subs(ss_sol_dict)
                                            for x in self.normal_x_s_tm1.values()]
                normal_x_s_tm1_ss_values_d = dict(
                    zip(self.normal_x_s_tm1.values(), normal_x_s_tm1_ss_values))

                normal_x_s_ss_values_d = {}
                normal_x_s_ss_values_d.update(normal_x_s_tp1_ss_values_d)
                normal_x_s_ss_values_d.update(normal_x_s_t_ss_values_d)
                normal_x_s_ss_values_d.update(normal_x_s_tm1_ss_values_d)

                normal_w_tp1_s_zero_pairs = [(x, 0)
                                             for x in self.normal_w_s_d.values()]
                normal_w_s_ss_values_d = dict(normal_w_tp1_s_zero_pairs)

                args_values_x = [x.subs(normal_x_s_ss_values_d)
                                 for x in self.normal_x_s_d.values()]

                args_values_w = [w.subs(normal_w_s_ss_values_d)
                                 for w in self.normal_w_s_d.values()]

                args_values_p = [p.subs(param_vals_d)
                                 for p in self.param_sym_dict.values()]

                args_values = args_values_x + args_values_w + args_values_p

        if mod == 'numpy':
            if self.fun_d_first_numpy == None or self.fun_d_second_numpy == None:
                self.fun_d_first_numpy, self.fun_d_second_numpy, vals = self.make_numpy_fns_of_d1d2xw(
                    d_g_dxw_1, d_g_dxw_2)
            d_first = self.fun_d_first_numpy(*args_values)
            d_second = self.fun_d_second_numpy(*args_values)

            d_first = [np.array(d, dtype='float') for d in d_first]
            d_second = [np.array(d, dtype='float') for d in d_second]
#            print '\nThis is get_evaluated_dgdxw12:'
#            print '\nOutput from make_numpy etc:'
#            print '\ntype(d_first):', type(d_first)
#            print '\ntype(d_first[0]):', type(d_first[0])
# print '\nd_first[0]):', d_first[0]
#            print '\ntype(d_first[0])):', type(d_first[0][0,0])
#
#            print '\ntype(d_second):', type(d_second)
#            print '\nlen(d_second):', len(d_second)
#            print '\ntype(d_second[0]):', type(d_second[0])
# print '\nd_second[0]):', d_second[0]
#            print '\ntype(d_second[0])):', type(d_second[0][0,0])

            return d_first, d_second

        elif mod == 'theano':
            if self.Need_compile_theano_fn_first:
                fn_d_first_th, fn_d_second_th, valth = self.make_theano_fns_of_d1d2xw(
                    d_g_dxw_1, d_g_dxw_2)
            d_first = fn_d_first_th(*args_values)
            d_second = fn_d_second_th(*args_values)
            return d_first, d_second

#        elif mod == 'theano':
#            if self.fun_d_first_theano == None or self.fun_d_second_theano == None:
#                self.fun_d_first_theano, self.fun_d_second_theano, valsth = self.make_theano_fns_of_d1d2xw(
#                    d_g_dxw_1, d_g_dxw_2)
#            d_first = self.fun_d_first_theano(*args_values)
#            d_second = self.fun_d_second_theano(*args_values)
#            return d_first, d_second

        else:
            print "Must specify 'numpy' or 'theano' "

    def d1d2_g_x_w_unevaluated(self):

        deriv_vars_x_t = self.xvar_t_sym
        deriv_vars_x_tm1 = self.xvar_tm1_sym
        deriv_vars_x_tp1 = self.xvar_tp1_sym
        deriv_vars_w_t = self.wvar_t_sym
        deriv_vars_w_tp1 = self.wvar_tp1_sym
        deriv_vars_q = [self.q]

        deriv_vars_all = [deriv_vars_x_tp1, deriv_vars_x_t, deriv_vars_x_tm1,
                          deriv_vars_w_tp1, deriv_vars_w_t, deriv_vars_q]

        g_eqs = sympy.Matrix(self.eq_conditions)

        d_g_d_xtp1 = g_eqs.jacobian(deriv_vars_x_tp1)
        d_g_d_xt = g_eqs.jacobian(deriv_vars_x_t)
        d_g_d_xtm1 = g_eqs.jacobian(deriv_vars_x_tm1)
        d_g_d_wtp1 = g_eqs.jacobian(deriv_vars_w_tp1)
        d_g_d_wt = g_eqs.jacobian(deriv_vars_w_t)
        d_g_d_q = g_eqs.jacobian(deriv_vars_q)

        d_g_first = [d_g_d_xtp1, d_g_d_xt, d_g_d_xtm1,
                     d_g_d_wtp1, d_g_d_wt,
                     d_g_d_q]

        d_g_second_stack = []

        for dg in d_g_first:
            for dvar in deriv_vars_all:
                sta_sec = [
                    dg[i, :].jacobian(dvar).T.vec().T for i in range(dg.rows)]
                totsec = sta_sec[0]
                for i in range(len(sta_sec)-1):
                    totsec = totsec.col_join(sta_sec[i+1])
#                print '\ntotsec.shape:', totsec.shape
                d_g_second_stack.append(totsec)

#        print '\nthis is d1d2_g_x_w_unevaluated:'
#        print '\nlen(d_g_second_stack):',len(d_g_second_stack)

        return d_g_first, d_g_second_stack

    def old_d1d2_g_x_w_unevaluated_old(self):

        deriv_vars_x_t = self.xvar_t_sym
        deriv_vars_x_tm1 = self.xvar_tm1_sym
        deriv_vars_x_tp1 = self.xvar_tp1_sym
        deriv_vars_w_t = self.wvar_t_sym
        deriv_vars_w_tp1 = self.wvar_tp1_sym

        deriv_vars_all = [deriv_vars_x_tp1, deriv_vars_x_t, deriv_vars_x_tm1,
                          deriv_vars_w_tp1, deriv_vars_w_t]

        g_eqs = sympy.Matrix(self.eq_conditions)

        d_g_d_xtp1 = g_eqs.jacobian(deriv_vars_x_tp1)
        d_g_d_xt = g_eqs.jacobian(deriv_vars_x_t)
        d_g_d_xtm1 = g_eqs.jacobian(deriv_vars_x_tm1)
        d_g_d_wtp1 = g_eqs.jacobian(deriv_vars_w_tp1)
        d_g_d_wt = g_eqs.jacobian(deriv_vars_w_t)

        d_g_first = [d_g_d_xtp1, d_g_d_xt, d_g_d_xtm1, d_g_d_wtp1, d_g_d_wt]

        d_g_second = []

        for dg in d_g_first:
            for dvar in deriv_vars_all:
                d2g = sympy.Matrix(list(dg)).jacobian(dvar)
                d_g_second.append(d2g)

        d_g_second_stack = []
        for dg in d_g_first:
            for dvar in deriv_vars_all:
                sta_sec = [
                    dg[i, :].jacobian(dvar).T.vec().T for i in range(dg.rows)]
                totsec = sta_sec[0]
                for i in range(len(sta_sec)-1):
                    totsec = totsec.col_join(sta_sec[i+1])
                print '\ntotsec.shape:', totsec.shape
                d_g_second_stack.append(totsec)


#        df_dx = feqsm.jacobian(xvars)
#        sta_sec = [df_dx[i,:].jacobian(wvars).T.vec().T for i in range(df_dx.rows)]
#        totsec = sta_sec[0]
#        for i in  range(len(sta_sec)-1):
#            totsec = totsec.col_join(sta_sec[i+1])

        print '\nthis is d1d2_g_x_w_unevaluated:'
        print '\nlen(d_g_second):', len(d_g_second)
        print '\nlen(d_g_second_stack):', len(d_g_second_stack)

        d_g_d_xtp1_d_xtp1 = sympy.Matrix(
            list(d_g_d_xtp1)).jacobian(deriv_vars_x_tp1)
        d_g_d_xtp1_d_xt = sympy.Matrix(
            list(d_g_d_xtp1)).jacobian(deriv_vars_x_t)
        d_g_d_xtp1_d_xtm1 = sympy.Matrix(
            list(d_g_d_xtp1)).jacobian(deriv_vars_x_tm1)
        d_g_d_xtp1_d_wtp1 = sympy.Matrix(
            list(d_g_d_xtp1)).jacobian(deriv_vars_w_tp1)
        d_g_d_xtp1_d_wt = sympy.Matrix(
            list(d_g_d_xtp1)).jacobian(deriv_vars_w_t)

        d_g_d_xt_d_xt = sympy.Matrix(list(d_g_d_xt)).jacobian(deriv_vars_x_t)
        d_g_d_xt_d_xtm1 = sympy.Matrix(
            list(d_g_d_xt)).jacobian(deriv_vars_x_tm1)
        d_g_d_xt_d_wtp1 = sympy.Matrix(
            list(d_g_d_xt)).jacobian(deriv_vars_w_tp1)
        d_g_d_xt_d_wt = sympy.Matrix(list(d_g_d_xt)).jacobian(deriv_vars_w_t)

        d_g_d_xtm1_d_xtm1 = sympy.Matrix(
            list(d_g_d_xtm1)).jacobian(deriv_vars_x_tm1)
        d_g_d_xtm1_d_wtp1 = sympy.Matrix(
            list(d_g_d_xtm1)).jacobian(deriv_vars_w_tp1)
        d_g_d_xtm1_d_wt = sympy.Matrix(
            list(d_g_d_xtm1)).jacobian(deriv_vars_w_t)

        d_g_d_wtp1_d_wtp1 = sympy.Matrix(
            list(d_g_d_wtp1)).jacobian(deriv_vars_w_tp1)
        d_g_d_wtp1_d_wt = sympy.Matrix(
            list(d_g_d_wtp1)).jacobian(deriv_vars_w_t)

        d_g_d_wt_d_wt = sympy.Matrix(list(d_g_d_wt)).jacobian(deriv_vars_w_t)

        d_g_firsts = [d_g_d_xtp1, d_g_d_xt, d_g_d_xtm1, d_g_d_wtp1, d_g_d_wt]

        d_g_seconds = [d_g_d_xtp1_d_xtp1, d_g_d_xtp1_d_xt, d_g_d_xtp1_d_xtm1,
                       d_g_d_xtp1_d_wtp1, d_g_d_xtp1_d_wt,
                       d_g_d_xt_d_xt, d_g_d_xt_d_xtm1, d_g_d_xt_d_wtp1,
                       d_g_d_xt_d_wt,
                       d_g_d_xtm1_d_xtm1, d_g_d_xtm1_d_wtp1, d_g_d_xtm1_d_wt,
                       d_g_d_wtp1_d_wtp1, d_g_d_wtp1_d_wt,
                       d_g_d_wt_d_wt]

        return d_g_firsts, d_g_seconds

    def make_numpy_fns_of_d1d2xw(self, dg_first, dg_second):

        args = self.normal_x_s_d.values() + self.normal_w_s_d.values() + \
            self.param_sym_dict.values()

        args_values_x = [x.subs(self.normal_xw_s_ss_values_d)
                         for x in self.normal_x_s_d.values()]

        args_values_w = [w.subs(self.normal_xw_s_ss_values_d)
                         for w in self.normal_w_s_d.values()]

        args_values_p = [p.subs(self.par_to_values_dict)
                         for p in self.param_sym_dict.values()]

        args_values = args_values_x + args_values_w + args_values_p

#        print '\nargs for lambdify:', args
#        print '\nargs values for lambdify:', args_values
#        print '\nself.normal_xw_s_ss_values_d:', self.normal_xw_s_ss_values_d
#        print '\nself.par_to_values_dict:', self.par_to_values_dict

        dg_first_lam = sympy.lambdify(args, dg_first, modules='numpy', dummify=False)
        dg_second_lam = sympy.lambdify(args, dg_second, modules='numpy', dummify=False)

        dtypes = {inp: 'float64' for inp in args}
        
#        print 'dtypes', dtypes
        
#        print 'end of print in inside make_theano_fns_of_d1d2xw:\n'
        dg_first_th = theano_function(args, dg_first,
                                      on_unused_input='ignore',
                                      dtypes=dtypes)
        
        dg_second_th = theano_function(
            args, dg_second, on_unused_input='ignore')



        return dg_first_lam, dg_second_lam, args_values
        

    def make_theano_fns_of_d1d2xw(self, dg_first, dg_second):

        args = self.normal_x_s_d.values() + self.normal_w_s_d.values() + \
            self.param_sym_dict.values()

        args_values_x = [x.subs(self.normal_xw_s_ss_values_d)
                         for x in self.normal_x_s_d.values()]

        args_values_w = [w.subs(self.normal_xw_s_ss_values_d)
                         for w in self.normal_w_s_d.values()]

        args_values_p = [p.subs(self.par_to_values_dict)
                         for p in self.param_sym_dict.values()]

        args_values = args_values_x + args_values_w + args_values_p

        print 'inside make_theano_fns_of_d1d2xw:\n'
        print 'args = ', args
        print 'dg_first: ', dg_first
        
        dtypes = {inp: 'float64' for inp in args}
        
        print 'dtypes', dtypes
        
        print 'end of print in inside make_theano_fns_of_d1d2xw:\n'
        dg_first_th = sympy.printing.theanocode.theano_function(args, dg_first,
                                      on_unused_input='ignore',
                                      dtypes=dtypes,
                                      mode='DebugMode')
        
        dg_second_th = sympy.printing.theanocode.theano_function(
            args, dg_second, on_unused_input='ignore')

        return dg_first_th, dg_second_th, args_values


    def get_first_order_approx_coeff_fi(self, eqs=[],
                                        param_vals_d={},
                                        mod='numpy'):

        beta = self.beta.subs(self.par_to_values_dict)
        theta = self.theta
        d_first, d_second = self.get_evaluated_dgdxw12(eqs, param_vals_d,
                                                       mod='numpy')

        gxtp1_ss, gx_ss, gxtm1_ss, gwtp1_ss, gwt_ss, gq_ss = d_first

        gxtp1_ss = matrix2numpyfloat(gxtp1_ss)
        gx_ss = matrix2numpyfloat(gx_ss)
        gxtm1_ss = matrix2numpyfloat(gxtm1_ss)
        gwtp1_ss = matrix2numpyfloat(gwtp1_ss)
        gwt_ss = matrix2numpyfloat(gwt_ss)

        nx = gxtp1_ss.shape[1]

#        print '\ngxtp1_ss:',gxtp1_ss
#        print '\ntype(gxtp1_ss):',type(gxtp1_ss)
#        print '\ntype(np.array(gxtp1_ss)):',type(np.array(gxtp1_ss))
        print '\nnx:', nx
        # quad_coeffmat_in_eq_for_P is Uhlig's \Psi matrix, sensible
        quad_coeffmat_in_eq_for_P = gxtp1_ss
        # lin_coeffmat_in_eq_for_P is Uhlig's (-\Gamma) matrix, sensible
        lin_coeffmat_in_eq_for_P = gx_ss
        # cons_coeffmat_in_eq_for_P is Uhlig's (-\Theta)\Psi matrix, sensible
        cons_coeffmat_in_eq_for_P = gxtm1_ss

        nnzero = np.zeros((nx, nx))

        # this is Uhlig's Chi matrix:
        lin_con_mat = np.bmat(
            [[-lin_coeffmat_in_eq_for_P, -cons_coeffmat_in_eq_for_P],
             [np.identity(nx), nnzero]])

        # this is Uhlig's Delta matrix:
        quad_mat = np.bmat(
            [[quad_coeffmat_in_eq_for_P, nnzero], [nnzero, np.identity(nx)]])

        stable_psi_x = self.solve_quad_matrix_stable_sol_QZ(lin_con_mat,
                                                            quad_mat, nx)

        psi_x = stable_psi_x
        psi_w_inner = np.dot(gxtp1_ss, stable_psi_x) + gx_ss
        psi_w = - np.dot(np.linalg.inv(psi_w_inner), gwt_ss)

#        psi_q part:
        psi_q_inv_term = scipy.linalg.inv(np.dot(gxtp1_ss, psi_x) +
                                          gxtp1_ss + gx_ss)

        coef_on_dist_E = np.dot(gxtp1_ss, psi_w) + gwtp1_ss

#        utility_ss_sym = self.utility.subs(self.ss_solutions_dict)

        utility_sym_mat = sympy.Matrix([self.utility])

        du_dx = utility_sym_mat.jacobian([self.xvar_t_sym])
        du_dx_nopar = [u.subs(self.par_to_values_dict) for u in list(du_dx)]
        du_dx_at_ss = [u.subs(self.normal_and_0_to_ss).subs(
            self.ss_solutions_dict) for u in du_dx_nopar]
#        print '\ndu_dx_at_ss:', du_dx_at_ss
        du_dx_at_ss = self.matrix2numpyfloat(sympy.Matrix(du_dx_at_ss))
        nx = len(self.xvar_t_sym)
        Ibetaphi = np.eye(nx) - beta*psi_x
        invIbetaphi = scipy.linalg.inv(Ibetaphi)
        print '\ndu_dx_at_ss', du_dx_at_ss
        print '\ndu_dx_at_ss.shape', du_dx_at_ss.shape
        print '\ndu_dx_at_ss.T.shape', du_dx_at_ss.T.shape
        print '\ninvIbetaphi.shape', invIbetaphi.shape
        Vx = np.dot(du_dx_at_ss.T, invIbetaphi)

        E_w_dist = - np.dot(Vx, psi_w).T/theta

        vec_for_diag = np.dot(coef_on_dist_E, E_w_dist)

#        print 'E_w_dist', E_w_dist
#        print 'coef_on_dist_E', coef_on_dist_E
#        print 'vec_for_diag', vec_for_diag
#
        diag_mat = np.diag(vec_for_diag)

        psi_q = - np.dot(psi_q_inv_term, gq_ss + diag_mat)

        return psi_x, psi_w, psi_q

    def get_second_order_approx_coeff_fi(self, psi_x, psi_w, d_first, d_second):

        gxtp1, gxt, gxtm1, gwtp1, gwpt, gq = d_first

        nx = psi_x.shape[1]
        nw = psi_w.shape[1]

        print '\nlen(d_second):', len(d_second)

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

    def solve_quad_matrix_stable_sol_QZ(self, lin_con_mat, quad_mat, nx):

        gqzresults = rgqz(np.asarray(lin_con_mat), np.asarray(quad_mat),
                          sort="S")

        # lin_con_mat is Uhlig's toolkit \Xi matrix
        # quad_mat is Uhlig's toolkit \Delta matrix

#        print '\n'
#        print 'type(lin_con_mat):' ,type(lin_con_mat)
#        print '\n'
#        print 'type(quad_mat):' ,type(quad_mat)


#        gqzresults = rgqz(np.asarray(lin_con_mat), np.asarray(quad_mat),
#                            sort="S")


#        S_s = np.mat(gqzresults_s[0])
#        T_s = np.mat(gqzresults_s[1])
#        Q_s = np.mat(gqzresults_s[6])
#        Z_s = np.mat(gqzresults_s[7])
#
        S = np.array(gqzresults[0])
        T = np.array(gqzresults[1])
        Q = np.array(gqzresults[6])
        Z = np.array(gqzresults[7])

#        nx = F_mat.shape[1]

        Z_11 = Z[0:nx, 0:nx]
        Z_12 = Z[0:nx, nx:]
        Z_21 = Z[nx:, 0:nx]
        Z_22 = Z[nx:, nx:]

#        P_mat = np.dot(Z_11, Z_21.I)

        P_mat = np.dot(Z_11, np.linalg.inv(Z_21))

        return P_mat

    def explain_phi_x(self):

        print '\n'
        print 'deriv_vars_x_t:'
        print self.xvar_t_sym

        print '\n'
        print 'deriv_vars_w_t:'
        print[self.wat_sym, self.wzetaat_sym]

    def impulse_response_toy_fi(self, phi_x, phi_w, shock=1, x0=[],
                                finalT=30, shock_type='wa', shock_mag=1,
                                dologs=False):

        #        print 'phi_x.shape',phi_x.shape
        #        print 'phi_x.shape[0]',phi_x.shape[0]

        steady_s_list_sym = [self.Css_sym, self.Kss_sym, self.Lss_sym,
                             self.Iss_sym, self.Yss_sym]

        steady_s_list_num = [x.subs(self.ss_solutions_dict) for x
                             in steady_s_list_sym]

        steady_s_vec = np.zeros((6, 1))
        steady_s_vec[0:-1, :] = np.array([steady_s_list_num]).T


#        print 'steady_s_list_sym', steady_s_list_sym
#        print 'steady_s_list_num', steady_s_list_num
#        print 'steady_s_vec', steady_s_vec
#

        if x0 == []:
            x0 = np.zeros((phi_x.shape[0], 1))

        x_time_series_mat = np.empty((x0.shape[0], finalT))

#        print '\n'
#        print 'x_time_series_mat.shape', x_time_series_mat.shape

        x_non_sta_time_series_mat = np.empty((x0.shape[0], finalT))

        azA_mat = np.empty((3, finalT))

        mua = self.mua
        zetaa_minusone = 0
        current_x = x0

        A_minusone = 1.0

        impulse_vector = np.zeros((2, 1))

        if shock_type == 'wa':
            impulse_vector[0, 0] = shock_mag
        elif shock_type == 'wzetaa':
            impulse_vector[1, 0] = shock_mag

        if dologs:
            linlinco = np.hstack((phi_x, phi_w))
            vss = self.get_steady_state_x_values()
            sel_vec = np.array([True, True, True, False, True, False,
                                False, False])
            v_allss_xw = np.vstack((vss, np.zeros((3, 1))))
            loglog_coeffs = self.from_linlin_to_loglog(linlinco, v_allss_xw,
                                                       sel_vec)
            phi_x = loglog_coeffs[:, 0:6]
            phi_w = loglog_coeffs[:, 6:8]
#            steady_s_vec = np.vstack((vss,np.zeros((1,1)))) #only x variables
            steady_s_vec[sel_vec] = np.log(steady_s_vec[sel_vec])

        first_response_x = np.dot(phi_x, x0) + np.dot(phi_w, impulse_vector)

#        print 'steady_s_vec line 942', steady_s_vec
#        print 'first_response_x',  first_response_x

        first_new_values_x = steady_s_vec + first_response_x

        x_time_series_mat[:, 0] = first_new_values_x[:, 0]
#        x_non_sta_time_series_mat[:,0] = first_response_x[:,0]/A_minusone
        x_non_sta_time_series_mat[:, 0] = first_new_values_x[:, 0]*A_minusone

        a_zero = np.dot(self.D, np.array( (mua, zetaa_minusone))) + \
            np.dot(self.G, impulse_vector)

        zetaa_zero = self.rhoa*zetaa_minusone + \
            np.dot(self.C, impulse_vector)[1, :]

        A_zero = A_minusone*np.exp(a_zero)

        azA_mat[0, 0] = a_zero  # a_0 has the direct impact of the impulse
        # z_{0} has the direct impact of the impulse
        azA_mat[1, 0] = zetaa_zero
        # A_{0} has the direct impact of the impulse
        azA_mat[2, 0] = A_minusone

        current_zetaa = zetaa_zero
        current_A = A_zero

        current_x = first_new_values_x  # C_0, K_1, L_0 etc.

#        print '\n'
#        print 'steady_s_vec', steady_s_vec

        for t in range(finalT-1):
            current_x = steady_s_vec + np.dot(phi_x, current_x-steady_s_vec)
            x_time_series_mat[:, t+1] = current_x[:, 0]  # C_1, K_2, L_1 ...
            # \tilde{C}_1, \tilde{K}_2, L_1 ...
#            x_non_sta_time_series_mat[:,t+1] = current_x[:,0]/current_A

            if dologs:
                x_non_sta_time_series_mat[
                    :, t+1] = current_x[:, 0] + np.log(current_A)
            else:
                x_non_sta_time_series_mat[:, t+1] = current_x[:, 0]*current_A

            # those are a_1 and z_0
            current_a = np.dot(self.D, np.array((mua, current_zetaa)))

            current_zetaa = self.rhoa*current_zetaa  # z_1 and z_0

            azA_mat[0, t+1] = current_a  # a_1, a_2 ...
            azA_mat[1, t+1] = current_zetaa  # z_1, z_2
            azA_mat[2, t+1] = current_A  # A_0, A_1

            current_A = current_A*np.exp(current_a)  # A_1, A_0, a_1

        x_non_sta_time_series_mat[2, :] = x_time_series_mat[2, :]

        x_non_sta_time_series_mat[-1, :] = x_time_series_mat[-1, :]

        BGP_det = np.zeros(x_non_sta_time_series_mat.shape)
        for i in range(BGP_det.shape[0]):
            BGP_det[i, :] = azA_mat[2, :]*steady_s_vec[i]

        return azA_mat, x_time_series_mat, x_non_sta_time_series_mat, BGP_det

    def matrix2numpyfloat(self, m):
        a = np.empty(m.shape, dtype='float')
        for i in range(m.rows):
            for j in range(m.cols):
                a[i, j] = m[i, j]
        return a

    def from_linlin_to_loglog(self, coeffs_of_linlin, ss_vec_xw, sel_vec):
        rowx = coeffs_of_linlin.shape[0]
        colx = coeffs_of_linlin.shape[1]

        ss_vec_x = ss_vec_xw[0:-2]

        coeffs_of_log_log = np.empty(coeffs_of_linlin.shape)

        for i in range(rowx):
            for j in range(colx):
                if sel_vec[i]:
                    valdenom = ss_vec_x[i]
                else:
                    valdenom = 1.0
                if sel_vec[j]:
                    valnum = ss_vec_xw[j]
                else:
                    valnum = 1.0
                ratio_ss = valnum/valdenom
                coeffs_of_log_log[i, j] = ratio_ss*coeffs_of_linlin[i, j]

        return coeffs_of_log_log
