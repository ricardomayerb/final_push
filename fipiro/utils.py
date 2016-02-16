import sympy
from scipy import linalg, optimize
import numpy as np
# from sympy.utilities.autowrap import ufuncify
# from sympy.printing.theanocode import theano_function
# import matplotlib.pyplot as plt

# if using R's geigen package to compute the ordered QZ decomposition, we
# need to load this package via rpy2:
# Note: Scipy 0.17 plans to pack its own ordered QZ
# import rpy2.robjects as robjects
# import rpy2.robjects.numpy2ri
# rpy2.robjects.numpy2ri.activate()
# robjects.r.library("geigen")
# rgqz = robjects.r.gqz


q = sympy.Symbol('q')
x_names = []
w_names = []
x_sym_dict = {}
xss_sym_dict = {}
w_sym_dict = {}
wss_sym_dict = {}

default_x_dates = ['tm1', 't', 'tp1']
default_w_dates = ['t', 'tp1']


def solve_quad_matrix_stable_sol_QZ(lin_con_mat, quad_mat, nx):

    gqzresults = rgqz(np.asarray(lin_con_mat), np.asarray(quad_mat),
                      sort="S")
    # lin_con_mat is Uhlig's toolkit \Xi matrix
    # quad_mat is Uhlig's toolkit \Delta matrix

    S = np.array(gqzresults[0])
    T = np.array(gqzresults[1])
    Q = np.array(gqzresults[6])
    Z = np.array(gqzresults[7])

    Z_11 = Z[0:nx, 0:nx]
    Z_12 = Z[0:nx, nx:]
    Z_21 = Z[nx:, 0:nx]
    Z_22 = Z[nx:, nx:]

    # # print "lin_con_mat", lin_con_mat
    # print "quad_mat", quad_mat

    # print "nx", nx

    # print "Z_21", Z_21

    P_mat = np.dot(Z_11, np.linalg.inv(Z_21))

    return P_mat


def get_sstate_sol_dict_from_sympy_eqs(glist, vars_ss_sym,
                                       vars_initvalues_dict={}):

    nx = len(glist)

    glist_lam = sympy.lambdify([vars_ss_sym], glist, dummify=False)

    if vars_initvalues_dict == {}:
        xini_list = 0.2 * np.ones(nx)
    else:
        xini_list = [x.subs(vars_initvalues_dict) for x in vars_ss_sym]

    xini_af = np.empty(len(xini_list), dtype='float')

    for i in range(len(xini_list)):
        xini_af[i] = xini_list[i]

    xini_list = np.array(xini_list)

    print "entering ss solver function"
    print "equations to solve:"
    print glist
    print "number of equations: " + str(nx) + "\n"
    print "list of variables: "
    print vars_ss_sym
    print "xini_list:"
    print xini_list
    print "xini_af:"
    print xini_af

    sol = optimize.root(glist_lam, xini_af)
    print "steady state solution\n"
    print sol
    print "\n end of ss solver function \n"

    return dict(zip(vars_ss_sym, np.around(sol['x'], decimals=12)))


def make_base_sym_dicts(x_names, w_names, param_names,
                        x_dates=default_x_dates, w_dates=default_w_dates):

    x_sym_dict = {}
    w_sym_dict = {}
    x_sym_dict = {}
    for t in x_dates:
        xvars_this_date = {x+t: sympy.Symbol(x+t) for x in x_names}
        x_sym_dict.update(xvars_this_date)

    for t in w_dates:
        wvars_this_date = {w+t: sympy.Symbol(w+t) for w in w_names}
        w_sym_dict.update(wvars_this_date)

    param_sym_dict = {k: sympy.Symbol(k) for k in param_names}

    x_sym_ss_dict = {k+'ss': sympy.Symbol(k+'ss') for k in x_names}
    w_sym_ss_dict = {k+'ss': sympy.Symbol(k+'ss') for k in w_names}
    wss_to_zero_dict = {wss: 0 for wss in w_sym_dict.values()}

    all_dicts = {'x': x_sym_dict, 'xss': x_sym_ss_dict, 'w': w_sym_dict,
                 'wss': w_sym_ss_dict, 'w_to_zero': wss_to_zero_dict,
                 'param': param_sym_dict}

    return all_dicts


def make_basic_sym_dict(names_list, date_string):
    """put docstring here"""
    dated_names = [name + date_string for name in names_list]
    sym_list = [sympy.Symbol(x) for x in dated_names]
    basic_sym_dic = dict(zip(dated_names, sym_list))
    return basic_sym_dic


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


def make_sym_dict(names_list, date_str, isw=False, do_derivatives=True):
    dic_to_be_filled = {}
    dated_names = [name + date_str for name in names_list]
    dated_names_sym = [sympy.Symbol(x) for x in dated_names]
    dic_to_be_filled.update(dict(zip(dated_names, dated_names_sym)))

    if isw:
        return dic_to_be_filled

    if do_derivatives:
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
        dated_q_mul_sym = [q * sympy.Symbol(x) for x in dated_names]
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


def make_sym_dict_all_dates(names_list, dates_list, isw=True,
                            do_steady_states_names=True):

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

    return x_sym_dict, w_sym_dict


def make_x_w_param_sym_dicts(this_x_names, this_w_names, this_param_names):
    x_sym_dict, w_sym_dict = set_x_w_sym_dicts(this_x_names, this_w_names)
    param_sym_dict = set_param_sym_dict(this_param_names)

    x_in_ss_sym_d = {st: x_sym_dict[st] for st in x_sym_dict.keys()
                     if 'ss' in st}

    return x_sym_dict, x_in_ss_sym_d, w_sym_dict, param_sym_dict


def make_state_space_sym(num_signals, num_states, is_homo):
    # mu_rho_dict_sym = {}
    state_state_sym_dict = {}

    # mu_names = ['mu_'+str(i) for i in range(num_states)]
    rho_names = ['rho_' + str(i) for i in range(num_states)]
    # mu_rho_names = mu_names + rho_names

    # mu_names_sym = [sympy.Symbol(x) for x in mu_names]
    rho_names_sym = [sympy.Symbol(x) for x in rho_names]
    # mu_rho_names_sym = [sympy.Symbol(x) for x in mu_rho_names]
    # mu_rho_dict_sym.update(dict(zip(mu_rho_names, mu_rho_names_sym)))

    A_identity_sym = sympy.eye(num_states)
    A_rhoblock_sym = sympy.diag(*rho_names_sym)
    A_sym = sympy.diag(A_identity_sym, A_rhoblock_sym)

    D_sym_mu_part = sympy.ones(num_signals, num_states)
    D_sym_zeta_part = sympy.ones(num_signals, num_states)
    D_sym = D_sym_mu_part.row_join(D_sym_zeta_part)

    if is_homo:
        sigmas_signal_names = [
            'sigma_signal_' + str(i) for i in range(num_signals)]
        sigmas_state_names = ['sigma_state_' + str(i) for i in
                              range(num_states)]
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


class BaseModel(object):

    def __init__(self, var_names_dict, param_names, equations=[],
                 param_num_dict={}, vars_initvalues_dict={},
                 var_dates_dict={'x_dates': ['tm1', 't', 'tp1'],
                                 'w_dates': ['t', 'tp1']}):

        self.equations = equations
        self.param_num_d = param_num_dict
        self.initvalues = vars_initvalues_dict

        self.var_names_d = var_names_dict

        if 'x_names' in var_names_dict.keys():
            self.x_names = var_names_dict['x_names']
        else:
            self.x_names = {}
        if 'w_names' in var_names_dict.keys():
            self.w_names = var_names_dict['w_names']
        else:
            self.w_names = {}

        if 'x_dates' in var_dates_dict.keys():
            self.x_dates = var_dates_dict['x_dates']
        else:
            self.x_dates = {}
        if 'w_dates' in var_dates_dict.keys():
            self.w_dates = var_dates_dict['w_dates']
        else:
            self.w_dates = {}

        base_dicts = make_base_sym_dicts(self.x_names, self.w_names,
                                         param_names, self.x_dates,
                                         self.w_dates)

        self.x_sym_d = base_dicts['x']
        self.w_sym_d = base_dicts['w']
        self.xss_sym_d = base_dicts['xss']
        self.wss_sym_d = base_dicts['wss']
        self.w_to_zero_sym_d = base_dicts['w_to_zero']
        self.param_sym_d = base_dicts['param']


class ModelBase(object):  # old class, with huge unit. BaseModel is newer
    """"docstring for """
    def __init__(self, equations, x_names, w_names,
                 x_dates=['tm1', 't', 'tp1'], w_dates=['t', 'tp1'],
                 param_names=[], par_to_values_dict={},
                 vars_initvalues_dict={}, compute_ss=True):
        self.eqns = equations
        self.x_names = x_names
        self.w_names = w_names
        self.param_names = param_names
        self.par_to_values_dict = par_to_values_dict

        self.x_dates = x_dates
        self.w_dates = w_dates

        xwp_sym_d = make_x_w_param_sym_dicts(
                x_names, w_names, param_names)

        x_s_d, x_in_ss_sym_d, w_s_d, param_sym_d = xwp_sym_d
        self.x_s_d = x_s_d
        self.x_in_ss_sym_d = x_in_ss_sym_d
        self.w_s_d = w_s_d
        self.param_sym_d = param_sym_d
        self.param_sym = list(sympy.ordered(param_sym_d.values()))

        self.x_in_ss_sym = list(sympy.ordered(self.x_in_ss_sym_d.values()))

        self.normal_x_s_d = {st: self.x_s_d[st] for st in self.x_s_d.keys()
                             if not ('_q' in st or '_1' in st or '_2' in st or
                             '_0' in st or 'ss' in st or 'q' in st)}

        self.normal_x_s_tp1 = {st: self.normal_x_s_d[st] for st in
                               self.normal_x_s_d.keys() if 'tp1' in st}

        self.normal_x_s_t = {st: self.normal_x_s_d[st] for st in
                             self.normal_x_s_d.keys() if not(
                             'tm1' in st or 'tp1' in st)}

        self.normal_x_s_tm1 = {st: self.normal_x_s_d[st] for st in
                               self.normal_x_s_d.keys() if 'tm1' in st}

        self.normal_w_s_d = {st: self.w_s_d[st] for st in self.w_s_d.keys() if
                             'ss' not in st}

        self.normal_w_s_tp1 = {st: self.normal_w_s_d[st] for st in
                               self.normal_w_s_d.keys() if 'tp1' in st}

        self.normal_w_s_t = {st: self.normal_w_s_d[st] for st in
                             self.normal_w_s_d.keys() if
                             not('tm1' in st or 'tp1' in st)}

        self.xvar_tp1_sym = self.normal_x_s_tp1.values()
        self.xvar_t_sym = self.normal_x_s_t.values()
        self.xvar_tm1_sym = self.normal_x_s_tm1.values()
        self.wvar_tp1_sym = self.normal_w_s_tp1.values()
        self.wvar_t_sym = self.normal_w_s_t.values()

        self.xvar_tp1_sym = list(sympy.ordered(self.xvar_tp1_sym))
        self.xvar_t_sym = list(sympy.ordered(self.xvar_t_sym))
        self.xvar_tm1_sym = list(sympy.ordered(self.xvar_tm1_sym))
        self.wvar_tp1_sym = list(sympy.ordered(self.wvar_tp1_sym))
        self.wvar_t_sym = list(sympy.ordered(self.wvar_t_sym))

        self.normal_and_0_to_ss = make_normal_to_steady_state(
            self.x_names, self.w_names)

        if compute_ss:
            eqns_no_param = self.make_ss_version_of_eqs(self.eqns)
            self.ss_solutions_dict = get_sstate_sol_dict_from_sympy_eqs(
                eqns_no_param, self.x_in_ss_sym,
                vars_initvalues_dict=vars_initvalues_dict)

            self.ss_residuals = [x.subs(self.ss_solutions_dict) for x in
                                 eqns_no_param]

            # print eqns_no_param

    def make_ss_version_of_eqs(self, eqs_list):
        """put docstring here"""
        eq_conditions_nopar = [
            x.subs(self.par_to_values_dict) for x in eqs_list]

        w_in_ss_zero_d = make_wss_to_zero_dict(self.w_names)

        eq_conditions_nopar_ss = [x.subs(self.normal_and_0_to_ss).subs(
            w_in_ss_zero_d) for x in eq_conditions_nopar]

        return eq_conditions_nopar_ss

    def shift_eqns_fwd(self, eqs_list, eqs_idx=[]):
        x_t_to_tp1_d = dict(zip(self.xvar_t_sym, self.xvar_tp1_sym))
        x_tm1_to_t_d = dict(zip(self.xvar_tm1_sym, self.xvar_t_sym))
        w_t_to_tp1_d = dict(zip(self.wvar_t_sym, self.wvar_tp1_sym))

        t_to_tp1_d = {}
        t_to_tp1_d.update(x_t_to_tp1_d)
        t_to_tp1_d.update(w_t_to_tp1_d)

        tm1_to_t_d = {}
        tm1_to_t_d.update(x_tm1_to_t_d)

        for i in eqs_idx:
            eqs_list[i] = eqs_list[i].subs(t_to_tp1_d)
            eqs_list[i] = eqs_list[i].subs(tm1_to_t_d)
        return eqs_list


class UhligModel(ModelBase):
    """docstring for UhligModel"""
    def __init__(self, equations, u_x_names=[], u_y_names=[], u_z_names=[],
                 x_names=[], w_names=[], param_names=[], block_indices={},
                 par_to_values_dict={}, fwd_shift_idx=[], aux_eqs=[],
                 vars_initvalues_dict={}, u_trans_dict={}, ss_sol_dict={}):

        print u_x_names
        print u_y_names
        print u_z_names
        print x_names
        print w_names

        ModelBase.__init__(self, equations, x_names, w_names, param_names,
                           par_to_values_dict=par_to_values_dict,
                           vars_initvalues_dict=vars_initvalues_dict,
                           compute_ss=False)

        u_x_tp1_sym_d = make_basic_sym_dic(u_x_names, 'tp1')
        u_x_t_sym_d = make_basic_sym_dic(u_x_names, 't')
        u_x_tm1_sym_d = make_basic_sym_dic(u_x_names, 'tm1')

        u_y_tp1_sym_d = make_basic_sym_dic(u_y_names, 'tp1')
        u_y_t_sym_d = make_basic_sym_dic(u_y_names, 't')

        u_z_tp1_sym_d = make_basic_sym_dic(u_z_names, 'tp1')
        u_z_t_sym_d = make_basic_sym_dic(u_z_names, 't')

        u_w_tp1_sym_d = make_basic_sym_dic(w_names, 'tp1')
        u_w_t_sym_d = make_basic_sym_dic(w_names, 't')

        self.u_x_tp1_sym = list(sympy.ordered(u_x_tp1_sym_d.values()))
        self.u_x_t_sym = list(sympy.ordered(u_x_t_sym_d.values()))
        self.u_x_tm1_sym = list(sympy.ordered(u_x_tm1_sym_d.values()))

        self.u_y_tp1_sym = list(sympy.ordered(u_y_tp1_sym_d.values()))
        self.u_y_t_sym = list(sympy.ordered(u_y_t_sym_d.values()))

        self.u_z_tp1_sym = list(sympy.ordered(u_z_tp1_sym_d.values()))
        self.u_z_t_sym = list(sympy.ordered(u_z_t_sym_d.values()))

        self.u_w_tp1_sym = list(sympy.ordered(u_w_tp1_sym_d.values()))
        self.u_w_t_sym = list(sympy.ordered(u_w_t_sym_d.values()))

        if fwd_shift_idx != []:
            equations = self.shift_eqns_fwd(equations, fwd_shift_idx)

        new_eqs = [x.subs(u_trans_dict) for x in equations]

        new_eqs.extend(aux_eqs)

        equations = new_eqs

        if ss_sol_dict == {}:
            eqns_no_param = self.make_ss_version_of_eqs(equations)
            self.ss_solutions_dict = get_sstate_sol_dict_from_sympy_eqs(
                eqns_no_param, self.x_in_ss_sym,
                vars_initvalues_dict=vars_initvalues_dict)

            self.ss_residuals = [x.subs(self.ss_solutions_dict) for x in
                                 eqns_no_param]

        self.block_indices = block_indices
        # print block_indices
        non_expec_idx = block_indices['non_expectational_block']
        expec_idx = block_indices['expectational_block']
        z_idx = block_indices['z_block']

        if x_names == []:
            x_names = u_x_names + u_y_names + u_z_names

        self.u_param_sym = self.param_sym_d.values()
        self.u_param_sym = list(sympy.ordered(self.u_param_sym))

        self.eqns = equations

        print "self.eqns"
        print self.eqns

        if isinstance(non_expec_idx, int):
            self.eqns_non_expec = [equations[non_expec_idx]]
        else:
            self.eqns_non_expec = [equations[i] for i in non_expec_idx]

        if isinstance(expec_idx, int):
            self.eqns_expec = [equations[expec_idx]]
        else:
            self.eqns_expec = [equations[i] for i in expec_idx]

        if isinstance(z_idx, int):
            self.eqns_z = [equations[z_idx]]
        else:
            self.eqns_z = [equations[i] for i in z_idx]

        print "\nself.eqns_non_expec"
        print self.eqns_non_expec
        print "\nself.eqns_expec"
        print self.eqns_expec
        print "\nself.eqns_z"
        print self.eqns_z

        self.x_to_devss_dict = make_devss_subs_dict(self.x_names,
                                                    self.x_dates)

        self.eqns_expec_devss = [x.subs(self.x_to_devss_dict) for x
                                 in self.eqns_expec]
        self.eqns_non_expec_devss = [x.subs(self.x_to_devss_dict) for x in
                                     self.eqns_non_expec]
        self.eqns_z_devss = [x.subs(self.x_to_devss_dict) for x in
                             self.eqns_z]

        print "\nself.eqns_non_expec_devss"
        print self.eqns_non_expec_devss
        print "\nself.eqns_expec_devss"
        print self.eqns_expec_devss
        print "\nself.eqns_z_devss"
        print self.eqns_z_devss

        self.jacobians_unev, self.jacobians_unev_ss = self.jacobians_sym_uh()

        self.uA_sym = self.jacobians_unev[0]
        self.uB_sym = self.jacobians_unev[1]
        self.uC_sym = self.jacobians_unev[2]
        self.uD_sym = self.jacobians_unev[3]
        self.uF_sym = self.jacobians_unev[4]
        self.uG_sym = self.jacobians_unev[5]
        self.uH_sym = self.jacobians_unev[6]
        self.uJ_sym = self.jacobians_unev[7]
        self.uK_sym = self.jacobians_unev[8]
        self.uL_sym = self.jacobians_unev[9]
        self.uM_sym = self.jacobians_unev[10]
        self.uN_sym = self.jacobians_unev[11]

        self.uA_sym_ss = self.jacobians_unev_ss[0]
        self.uB_sym_ss = self.jacobians_unev_ss[1]
        self.uC_sym_ss = self.jacobians_unev_ss[2]
        self.uD_sym_ss = self.jacobians_unev_ss[3]
        self.uF_sym_ss = self.jacobians_unev_ss[4]
        self.uG_sym_ss = self.jacobians_unev_ss[5]
        self.uH_sym_ss = self.jacobians_unev_ss[6]
        self.uJ_sym_ss = self.jacobians_unev_ss[7]
        self.uK_sym_ss = self.jacobians_unev_ss[8]
        self.uL_sym_ss = self.jacobians_unev_ss[9]
        self.uM_sym_ss = self.jacobians_unev_ss[10]
        self.uN_sym_ss = self.jacobians_unev_ss[11]

        args = self.x_in_ss_sym + self.u_param_sym
        self.jac_ss_funcs = sympy.lambdify(args, self.jacobians_unev_ss)

        x_ss_num = [x.subs(self.ss_solutions_dict) for x in self.x_in_ss_sym]
        par_ss_num = [x.subs(self.par_to_values_dict) for x in self.param_sym]
        x_par_ss_num = x_ss_num + par_ss_num
        self.jac_ss_num = [matrix2numpyfloat(x) for x
                           in self.jac_ss_funcs(*x_par_ss_num)]
        # # print '\nself.jac_ss_num'
        # print self.jac_ss_num

        self.uA_num_ss = self.jac_ss_num[0]
        self.uB_num_ss = self.jac_ss_num[1]
        self.uC_num_ss = self.jac_ss_num[2]
        self.uD_num_ss = self.jac_ss_num[3]
        # print self.uD_sym
        # print self.uD_sym_ss
        # print self.uD_num_ss
        self.uF_num_ss = self.jac_ss_num[4]
        self.uG_num_ss = self.jac_ss_num[5]
        self.uH_num_ss = self.jac_ss_num[6]
        self.uJ_num_ss = self.jac_ss_num[7]
        self.uK_num_ss = self.jac_ss_num[8]
        self.uL_num_ss = self.jac_ss_num[9]
        self.uM_num_ss = self.jac_ss_num[10]
        self.uN_num_ss = self.jac_ss_num[11]

        self.u_x_ss_sym = [x.subs(self.normal_and_0_to_ss) for x in self.u_x_t_sym]
        self.u_y_ss_sym = [x.subs(self.normal_and_0_to_ss) for x in self.u_y_t_sym]
        self.u_z_ss_sym = [x.subs(self.normal_and_0_to_ss) for x in self.u_z_t_sym]

        self.u_x_ss_num = [x.subs(self.ss_solutions_dict) for x in self.u_x_ss_sym]
        self.u_y_ss_num = [x.subs(self.ss_solutions_dict) for x in self.u_y_ss_sym]
        self.u_z_ss_num = [x.subs(self.ss_solutions_dict) for x in self.u_z_ss_sym]
        # self.u_z_ss_num = [0.03, 0.03, 1]
        # self.u_z_ss_num = [0.03, 1]

        non_zero_z_idx = np.nonzero(self.u_z_ss_num)
        z_for_diag = np.ones_like(self.u_z_ss_num)
        z_for_diag[non_zero_z_idx] = self.u_z_ss_num

        print "foooooo\n"
        print self.u_z_ss_num
        print non_zero_z_idx
        print z_for_diag
        print "moooooo\n"


        self.di_u_x_ss_sym = sympy.diag(*self.u_x_ss_sym)
        self.di_u_y_ss_sym = sympy.diag(*self.u_y_ss_sym)
        self.di_u_z_ss_sym = sympy.diag(*self.u_z_ss_sym)

        self.di_u_x_ss_num = np.diag(self.u_x_ss_num)
        self.di_u_y_ss_num = np.diag(self.u_y_ss_num)
        self.di_u_z_ss_num = np.diag(z_for_diag)

        self.uA_num_ss_log = np.dot(self.uA_num_ss, self.di_u_x_ss_num)
        self.uB_num_ss_log = np.dot(self.uB_num_ss, self.di_u_x_ss_num)
        self.uC_num_ss_log = np.dot(self.uC_num_ss, self.di_u_y_ss_num)
        self.uD_num_ss_log = np.dot(self.uD_num_ss, self.di_u_z_ss_num)
        self.uF_num_ss_log = np.dot(self.uF_num_ss, self.di_u_x_ss_num)
        self.uG_num_ss_log = np.dot(self.uG_num_ss, self.di_u_x_ss_num)
        self.uH_num_ss_log = np.dot(self.uH_num_ss, self.di_u_x_ss_num)
        self.uJ_num_ss_log = np.dot(self.uJ_num_ss, self.di_u_y_ss_num)
        self.uK_num_ss_log = np.dot(self.uK_num_ss, self.di_u_y_ss_num)
        self.uL_num_ss_log = np.dot(self.uL_num_ss, self.di_u_z_ss_num)
        self.uM_num_ss_log = np.dot(self.uM_num_ss, self.di_u_z_ss_num)
        self.uN_num_ss_log = np.dot(self.uN_num_ss, self.di_u_z_ss_num)

    def transform_eqs_uh(self, eqs_list, var_subs_d):
        """put docstring here"""
        eqns_new_vars = [x.subs(var_subs_d) for x in eqs_list]

        return eqns_new_vars

    def jacobians_sym_uh(self, options={}):
        """put docstring here"""
        g_nonexp_eqns = sympy.Matrix(self.eqns_non_expec)
        g_exp_eqns = sympy.Matrix(self.eqns_expec)
        g_z_eqns = sympy.Matrix(self.eqns_z)

        d_gne_d_xt = g_nonexp_eqns.jacobian(self.u_x_t_sym)
        d_gne_d_xtm1 = g_nonexp_eqns.jacobian(self.u_x_tm1_sym)
        d_gne_d_yt = g_nonexp_eqns.jacobian(self.u_y_t_sym)
        d_gne_d_zt = g_nonexp_eqns.jacobian(self.u_z_t_sym)

        d_ge_d_xtp1 = g_exp_eqns.jacobian(self.u_x_tp1_sym)
        d_ge_d_xt = g_exp_eqns.jacobian(self.u_x_t_sym)
        d_ge_d_xtm1 = g_exp_eqns.jacobian(self.u_x_tm1_sym)
        d_ge_d_ytp1 = g_exp_eqns.jacobian(self.u_y_tp1_sym)
        d_ge_d_yt = g_exp_eqns.jacobian(self.u_y_t_sym)
        d_ge_d_ztp1 = g_exp_eqns.jacobian(self.u_z_tp1_sym)
        d_ge_d_zt = g_exp_eqns.jacobian(self.u_z_t_sym)

        d_gz_d_zt = g_z_eqns.jacobian(self.u_z_t_sym)

        # d_gne_d_xt_ss = d_gne_d_xt.subs(self.normal_and_0_to_ss)
        # print d_gne_d_xt_ss
        d_g_uhlig_unev = [d_gne_d_xt, d_gne_d_xtm1,
                          d_gne_d_yt, d_gne_d_zt,
                          d_ge_d_xtp1, d_ge_d_xt, d_ge_d_xtm1,
                          d_ge_d_ytp1, d_ge_d_yt,
                          d_ge_d_ztp1, d_ge_d_zt,
                          d_gz_d_zt]
        d_g_uhlig_unev_ss = [x.subs(self.normal_and_0_to_ss) for x in
                             list(d_g_uhlig_unev)]
        return [d_g_uhlig_unev, d_g_uhlig_unev_ss]
