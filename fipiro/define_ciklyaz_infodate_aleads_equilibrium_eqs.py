import sympy

from fipiro import utils as u

x_names = ['a', 'C', 'I', 'K', 'L', 'Y', 'zetaa']
x_dates = ['tm1', 't', 'tp1']
w_names = ['wa', 'wzetaa']
w_dates = ['t', 'tp1']
param_names = ['alpha', 'beta', 'delta', 'mua',
               'rhoa', 'nu', 'sigmaa', 'sigmazetaa']
all_names = {
    'x_names': x_names, 'w_names': w_names, 'param_names': param_names}

param_names_module_to_user = {'rho_0': 'rhoa', 'mu_0': 'mua',
                              'sigma_signal_0': 'sigmaa',
                              'sigma_state_0': 'sigmazetaa'}

xxsswp_sym_d = u.make_x_w_param_sym_dicts(x_names, w_names, param_names)

x_s_d, x_in_ss_sym_d, w_s_d, param_sym_d = xxsswp_sym_d


for k, v in x_s_d.iteritems():
    exec(k + '= v')

for k, v in w_s_d.iteritems():
    exec(k + '= v')

for k, v in param_sym_d.iteritems():
    exec(k + '= v')


def eq_conditions_TFP_FI_CIKLY_az_usedating_aleads():

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
