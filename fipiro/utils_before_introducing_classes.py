import sympy
from scipy import linalg, optimize
import numpy as np
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


def shift_forward_t_eqs(eqs_list, tm1_dic, t_dic, tp1_dic):
    # for equations where t is leading, shif it to tp1
    t_shifted = [x.subs({t:tp1}) for x in eqs_list]
    tm1_shifted = [x.subs({tm1:t}) for x in eqs_list]

    return tm1_shifted




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

    # print "lin_con_mat", lin_con_mat
    # print "quad_mat", quad_mat

    # print "nx", nx

    # print "Z_21", Z_21

    P_mat = np.dot(Z_11, np.linalg.inv(Z_21))

    return P_mat


def get_sstate_sol_dict_from_sympy_eqs(glist, xss_sym_dict, xini_dict={}):

    nx = len(glist)

    glist_lam = sympy.lambdify([xss_sym_dict.values()], glist, dummify=False)

    glist_lam

    if xini_dict == {}:
        xini_list = 0.2 * np.ones(nx)
    else:
        xini_list = [x.subs(xini_dict) for x in xss_sym_dict.values()]

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

def make_xw_tm1ttp1_dicts(x_names, w_names):

    x_sym_tm1_dict = make_sym_dict(x_names, 'tm1')
    x_sym_t_dict = make_sym_dict(x_names, 't')
    x_sym_tp1_dict = make_sym_dict(x_names, 'tp1')
    w_sym_t_dict = make_sym_dict(w_names, 't')
    w_sym_tp1_dict = make_sym_dict(w_names, 'tp1')

    return [x_sym_tm1_dict, x_sym_t_dict, x_sym_tp1_dict,
            w_sym_t_dict, w_sym_tp1_dict]


def make_x_w_param_sym_dicts(this_x_names, this_w_names, this_param_names):
    x_sym_dict, w_sym_dict = set_x_w_sym_dicts(this_x_names, this_w_names)
    param_sym_dict = set_param_sym_dict(this_param_names)

    x_in_ss_sym_d = {st: x_sym_dict[st] for st in x_sym_dict.keys()
                     if 'ss' in st}

    return x_sym_dict, x_in_ss_sym_d, w_sym_dict, param_sym_dict


def Fckliy_az_mua(xini_az):
    barC, barK, barL, barI, barY, bara, barzetaa = xini_az[0], xini_az[
        1], xini_az[2], xini_az[3], xini_az[4], xini_az[5], xini_az[6]

    foc1 = beta * np.exp(-mua) * (alpha * barY / barK + 1 - delta) - 1
    foc2 = barY - barC - barI
    foc3 = barI - np.exp(mua) * barK + (1 - delta) * barK
    nurat = nu / (1 - nu)
    foc4 = nurat * (1 - alpha) * barY / barC - barL / (1 - barL)
    foc5 = barY - np.exp((1 - alpha) * mua) * barK**alpha * barL**(1 - alpha)
    foc6 = bara - mua - barzetaa
    foc7 = barzetaa - rhoavalue * barzetaa
    focsckliy = np.array([foc1, foc2, foc3, foc4, foc5, foc6, foc7])

    return focsckliy


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
        sigmas_state_names = ['sigma_state_' + str(i) for i in range(num_states)]
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


# def all_vars_par_par(a_of_values, set_par_x, set_par_lev, par_tex_x,
#     par_tex_lev, par_name_x, par_name_lev, vars_tex, postname):
#
#     for v in range(len(set_par_lev)):
#         fig, ax = plt.subplots()
#         for i in range(5):
#             ax.plot(set_par_x, a_of_values[i, v, :], label=vars_tex[i])
#
#         ax.legend(loc="best", ncol=2)
#         ax.grid(True)
#         ax.set_title("Coeff of persistent shock " +
#                      "("+par_tex_lev + " = " +
#                      "{0:.2f})".format(set_par_lev[v]))
#         ax.set_xlabel(par_tex_x)
#         filename = "../../figures/psi_change_" + par_name_x + "_" + \
#             par_name_lev + "_all_vars_" + str(v) + "_" + postname + ".pdf"
#         fig.savefig(filename)
#

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
