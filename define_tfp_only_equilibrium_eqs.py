import sympy
from sympy.utilities.lambdify import lambdify
from math import exp

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


def eq_conds_TFP_FI_CIKLY_az_usedating_aleads_sta_all():

	It_expr = sympy.exp(at) * Ktp1 - (1-delta)*Kt
	Yt_expr = sympy.exp((1-alpha)*at) * Kt**(alpha) * Lt**(1-alpha)
	Ytp1_expr = sympy.exp((1-alpha)*atp1) * Ktp1**(alpha) * Ltp1**(1-alpha)

	at_expr = mua + zetaatm1 + sigmaa*wat
	atp1_expr = mua + zetaat + sigmaa*watp1

	zetaat_expr = rhoa*zetaatm1 + sigmazetaa*wzetaat

	g1 = beta*sympy.exp(-at) * (Ct/Ctp1) * \
		(alpha * Ytp1/Ktp1 + (1-delta)) - 1

	g2 = nu*(1-alpha)*Yt/(Ct*(1-nu)) - Lt/(1-Lt)

	g3 = Yt - It - Ct

	g4 = It - It_expr

	g5 = Yt - Yt_expr

	g6 = zetaat - zetaat_expr

	g7 = at - at_expr

	glist = [g1, g2, g3, g4, g5, g6, g7]
	invout_dict = {Yt: Yt_expr,
			   Ytp1: Ytp1_expr,
			   It: It_expr}
	az_dict = {at: at_expr, atp1: atp1_expr}
	return glist, az_dict, invout_dict


def eq_conds_TFP_FI_CIKLY_az_infodating_aleads_sta_all():

	It_expr = Kt - (1-delta) * sympy.exp(-atm1) * Ktm1
	Yt_expr = sympy.exp((1-alpha)*at - alpha*atm1) * Ktm1**(alpha) * Lt**(1-alpha)
	Ytp1_expr = sympy.exp((1-alpha)*atp1 - alpha*at) * Kt**(alpha) * Ltp1**(1-alpha)

	at_expr = mua + zetaatm1 + sigmaa*wat
	atp1_expr = mua + zetaat + sigmaa*watp1

	zetaat_expr = rhoa*zetaatm1 + sigmazetaa*wzetaat

	g1 = beta * (Ct/Ctp1) * \
		(alpha * Ytp1/Kt + (1-delta)* sympy.exp(-at)) - 1

	g2 = nu*(1-alpha)*Yt/(Ct*(1-nu)) - Lt/(1-Lt)

	g3 = Yt - It - Ct

	g4 = It - It_expr

	g5 = Yt - Yt_expr

	g6 = zetaat - zetaat_expr

	g7 = at - at_expr

	glist = [g1, g2, g3, g4, g5, g6, g7]
	invout_dict = {Yt: Yt_expr,
			   Ytp1: Ytp1_expr,
			   It: It_expr}
	az_dict = {at: at_expr, atp1: atp1_expr}
	return glist, az_dict, invout_dict


def eq_conds_TFP_FI_CIKLY_az_usedating_aconte_sta_all():

	It_expr = sympy.exp(at) * Ktp1 - (1-delta)*Kt
	Yt_expr = sympy.exp((1-alpha)*at) * Kt**(alpha) * Lt**(1-alpha)
	Ytp1_expr = sympy.exp((1-alpha)*atp1) * Ktp1**(alpha) * Ltp1**(1-alpha)

	at_expr = mua + zetaat + sigmaa*wat
	atp1_expr = mua + zetaatp1 + sigmaa*watp1

	zetaat_expr = rhoa*zetaatm1 + sigmazetaa*wzetaat

	g1 = beta*sympy.exp(-at) * (Ct/Ctp1) * \
		(alpha * Ytp1/Ktp1 + (1-delta)) - 1

	g2 = nu*(1-alpha)*Yt/(Ct*(1-nu)) - Lt/(1-Lt)

	g3 = Yt - It - Ct

	g4 = It - It_expr

	g5 = Yt - Yt_expr

	g6 = zetaat - zetaat_expr

	g7 = at - at_expr

	glist = [g1, g2, g3, g4, g5, g6, g7]
	invout_dict = {Yt: Yt_expr,
			   Ytp1: Ytp1_expr,
			   It: It_expr}
	az_dict = {at: at_expr, atp1: atp1_expr}
	return glist, az_dict, invout_dict


def eq_conds_TFP_FI_CIKLY_az_infodating_aconte_sta_all():

	It_expr = Kt - (1-delta) * sympy.exp(-atm1) * Ktm1
	Yt_expr = sympy.exp((1-alpha)*at - alpha*atm1) * Ktm1**(alpha) * Lt**(1-alpha)
	Ytp1_expr = sympy.exp((1-alpha)*atp1 - alpha*at) * Kt**(alpha) * Ltp1**(1-alpha)

	at_expr = mua + zetaat + sigmaa*wat
	atp1_expr = mua + zetaatp1 + sigmaa*watp1

	zetaat_expr = rhoa*zetaatm1 + sigmazetaa*wzetaat

	g1 = beta * (Ct/Ctp1) * \
		(alpha * Ytp1/Kt + (1-delta)* sympy.exp(-at)) - 1

	g2 = nu*(1-alpha)*Yt/(Ct*(1-nu)) - Lt/(1-Lt)

	g3 = Yt - It - Ct

	g4 = It - It_expr

	g5 = Yt - Yt_expr

	g6 = zetaat - zetaat_expr

	g7 = at - at_expr

	glist = [g1, g2, g3, g4, g5, g6, g7]
	invout_dict = {Yt: Yt_expr,
			   Ytp1: Ytp1_expr,
			   It: It_expr}
	az_dict = {at: at_expr, atp1: atp1_expr}
	return glist, az_dict, invout_dict


def ss_sym_sols_TFP_FI_CIKLY_az_usedating_aleads_sta_all():
	sol_zetaa_ss = 0
	sol_a_ss = mua
	theta = nu * (1-alpha)/(1-nu)

	B1 =  beta * (1-delta) * sympy.exp(-sol_a_ss)
	B2 = (1-B1)/(alpha * beta * sympy.exp(-sol_a_ss))
	B3 = sympy.exp(sol_a_ss) - (1-delta)
	B4 = 1/(B2 - B3)
	B5 = theta * B2 * B4
	B6 = B5/(1+B5)
	B7 = sympy.exp((1-alpha)*sol_a_ss)

	sol_L_ss = B6
	sol_K_ss = (B7/B2)**(1/(1-alpha)) * sol_L_ss
	sol_I_ss = sol_K_ss * B3
	sol_C_ss = sol_K_ss * (B2-B3)
	sol_Y_ss = B7 * sol_K_ss**alpha * sol_L_ss**(1-alpha)
	return [sol_C_ss, sol_I_ss, sol_K_ss, sol_L_ss, sol_Y_ss, sol_a_ss, sol_zetaa_ss]

def ss_eq_conds_TFP_FI_CIKLY_az_usedating_aleads_sta_all(sols_ciklyaz, pardict):
	Css = sols_ciklyaz[0]
	Iss = sols_ciklyaz[1]
	Kss = sols_ciklyaz[2]
	Lss = sols_ciklyaz[3]
	Yss = sols_ciklyaz[4]
	ass = sols_ciklyaz[5]
	zetaass = sols_ciklyaz[6]
	nu = pardict['nu']
	alpha = pardict['alpha']
	beta = pardict['beta']
	delta = pardict['delta']
	mua = pardict['mua']
	rhoa = pardict['rhoa']

	theta = nu * (1-alpha)/(1-nu)

	Iss_expr = exp(ass) * Kss - (1-delta)*Kss
	Yss_expr = exp((1-alpha)*ass) * Kss**(alpha) * Lss**(1-alpha)

	zetaass_expr = rhoa*zetaass
	ass_expr = mua + zetaass

	g1 = beta*exp(-ass)  * (alpha * Yss/Kss + (1-delta)) - 1

	g2 = nu*(1-alpha)*Yss/(Css*(1-nu)) - Lss/(1-Lss)

	g3 = Yss - Iss - Css

	g4 = Iss - Iss_expr

	g5 = Yss - Yss_expr

	g6 = zetaass - zetaass_expr

	g7 = ass - ass_expr

	glist = [g1, g2, g3, g4, g5, g6, g7]
	return glist

def ss_sym_sols_TFP_FI_CIKLY_az_infodating_aleads_sta_all():
	sol_zetaa_ss = 0
	sol_a_ss = mua
	theta = nu * (1-alpha)/(1-nu)

	B1 =  beta * (1-delta) * sympy.exp(-sol_a_ss)
	B2 = (1-B1)/(alpha * beta)
	B3 = 1 - (1-delta) * sympy.exp(-sol_a_ss)
	B4 = 1/(B2 - B3)
	B5 = theta * B2 * B4
	B6 = B5/(1+B5)
	B7 = sympy.exp((1-alpha)*sol_a_ss - alpha*sol_a_ss)

	sol_L_ss = B6
	sol_K_ss = (B7/B2)**(1/(1-alpha)) * sol_L_ss
	sol_I_ss = sol_K_ss * B3
	sol_C_ss = sol_K_ss * (B2-B3)
	sol_Y_ss = B7 * sol_K_ss**alpha * sol_L_ss**(1-alpha)

	return [sol_C_ss, sol_I_ss, sol_K_ss, sol_L_ss, sol_Y_ss, sol_a_ss, sol_zetaa_ss]

def ss_eq_conds_TFP_FI_CIKLY_az_infodating_aleads_sta_all(sols_ciklyaz, pardict):
	Css = sols_ciklyaz[0]
	Iss = sols_ciklyaz[1]
	Kss = sols_ciklyaz[2]
	Lss = sols_ciklyaz[3]
	Yss = sols_ciklyaz[4]
	ass = sols_ciklyaz[5]
	zetaass = sols_ciklyaz[6]

	nu = pardict['nu']
	alpha = pardict['alpha']
	beta = pardict['beta']
	delta = pardict['delta']
	mua = pardict['mua']
	rhoa = pardict['rhoa']

	theta = nu * (1-alpha)/(1-nu)


	Iss_expr = Kss - (1-delta) * exp(-ass) * Kss
	Yss_expr = exp((1-alpha)*ass - alpha*ass) * Kss**(alpha) * Lss**(1-alpha)

	ass_expr = mua + zetaass

	zetaass_expr = rhoa*zetaass

	g1 = beta * (alpha * Yss/Kss + (1-delta)* exp(-ass)) - 1

	g2 = nu*(1-alpha)*Yss/(Css*(1-nu)) - Lss/(1-Lss)

	g3 = Yss - Iss - Css

	g4 = Iss - Iss_expr

	g5 = Yss - Yss_expr

	g6 = zetaass - zetaass_expr

	g7 = ass - ass_expr

	glist = [g1, g2, g3, g4, g5, g6, g7]

	return glist


glist_cikly_az_info, iy1, az1 = eq_conds_TFP_FI_CIKLY_az_infodating_aleads_sta_all()
glist_cikly_az_use, iy2, az2 = eq_conds_TFP_FI_CIKLY_az_usedating_aleads_sta_all()
glist_cikly_az_info_conte, iy3, az3 = eq_conds_TFP_FI_CIKLY_az_infodating_aconte_sta_all()
glist_cikly_az_use_conte, iy4, az4 = eq_conds_TFP_FI_CIKLY_az_usedating_aconte_sta_all()

symsols_info = ss_sym_sols_TFP_FI_CIKLY_az_infodating_aleads_sta_all()
solC_i, solI_i, solK_i, solL_i, solY_i, sola_i, solzetaa_i = symsols_info

symsols_use = ss_sym_sols_TFP_FI_CIKLY_az_usedating_aleads_sta_all()
solC_u, solI_u, solK_u, solL_u, solY_u, sola_u, solzetaa_u = symsols_use


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

pref_tech_vars_to_values = {alpha: alpha_value, beta: beta_value,
							 delta: delta_value, nu: nu_value}

ssp_par_user_to_values = {mua: mua_value_true_Toy,
						  rhoa: rhoa_value,
						  sigmaa: sigmaa_value,
						  sigmazetaa: sigmaza_value}
all_param_values_dict = {}
all_param_values_dict.update(pref_tech_vars_to_values)
all_param_values_dict.update(ssp_par_user_to_values)

pref_tech_names_to_values = {'alpha': alpha_value, 'beta': beta_value,
                             'delta': delta_value, 'nu': nu_value}

ssp_par_user_names_to_values = {'mua': mua_value_true_Toy,
                          'rhoa': rhoa_value,
                          'sigmaa': sigmaa_value,
                          'sigmazetaa': sigmaza_value}

all_param_values_names_dict = {}
all_param_values_names_dict.update(pref_tech_names_to_values)
all_param_values_names_dict.update(ssp_par_user_names_to_values)

solC_i.subs({alpha:0.36, mua:0.03, beta:0.98, delta:0.1, nu:0.29})

sol_CIKLYaz_ss_info_lam = lambdify(all_param_values_dict.keys(),
						   [solC_i, solI_i,solK_i,solL_i,solY_i,sola_i,solzetaa_i])

num_sols_CIKLYaz_ss_info = sol_CIKLYaz_ss_info_lam(*all_param_values_dict.values())

sol_CIKLYaz_ss_use_lam = lambdify(all_param_values_dict.keys(),
						   [solC_u, solI_u,solK_u,solL_u,solY_u,sola_u,solzetaa_u])

num_sols_CIKLYaz_ss_use= sol_CIKLYaz_ss_use_lam(*all_param_values_dict.values())


resid_CIKLYaz_ss_info = ss_eq_conds_TFP_FI_CIKLY_az_infodating_aleads_sta_all(num_sols_CIKLYaz_ss_info, all_param_values_names_dict)

resid_CIKLYaz_ss_use = ss_eq_conds_TFP_FI_CIKLY_az_usedating_aleads_sta_all(num_sols_CIKLYaz_ss_use, all_param_values_names_dict)



glist_cikly_z_info = [x.subs(az1) for x in glist_cikly_az_info]
glist_cikly_z_info = [x for x in glist_cikly_z_info if x != 0]
glist_ckl_az_info = [x.subs(iy1) for x in glist_cikly_az_info]
glist_ckl_az_info = [x for x in glist_ckl_az_info if x != 0]
glist_ckl_z_info = [x.subs(az1) for x in glist_ckl_az_info]
glist_ckl_z_info = [x for x in glist_ckl_z_info if x != 0]

glist_cikly_z_use = [x.subs(az2) for x in glist_cikly_az_use]
glist_cikly_z_use = [x for x in glist_cikly_z_use if x != 0]
glist_ckl_az_use = [x.subs(iy2) for x in glist_cikly_az_use]
glist_ckl_az_use = [x for x in glist_ckl_az_use if x != 0]
glist_ckl_z_use = [x.subs(az2) for x in glist_ckl_az_use]
glist_ckl_z_use = [x for x in glist_ckl_z_use if x != 0]

glist_cikly_z_info_conte = [x.subs(az3) for x in glist_cikly_az_info_conte]
glist_cikly_z_info_conte = [x for x in glist_cikly_z_info_conte if x != 0]
glist_ckl_az_info_conte = [x.subs(iy3) for x in glist_cikly_az_info_conte]
glist_ckl_az_info_conte = [x for x in glist_ckl_az_info_conte if x != 0]
glist_ckl_z_info_conte = [x.subs(az3) for x in glist_ckl_az_info_conte]
glist_ckl_z_info_conte = [x for x in glist_ckl_z_info_conte if x != 0]

glist_cikly_z_use_conte = [x.subs(az4) for x in glist_cikly_az_use_conte]
glist_cikly_z_use_conte = [x for x in glist_cikly_z_use_conte if x != 0]
glist_ckl_az_use_conte = [x.subs(iy4) for x in glist_cikly_az_use_conte]
glist_ckl_az_use_conte = [x for x in glist_ckl_az_use_conte if x != 0]
glist_ckl_z_use_conte = [x.subs(az4) for x in glist_ckl_az_use_conte]
glist_ckl_z_use_conte = [x for x in glist_ckl_z_use_conte if x != 0]

######## uligh part
