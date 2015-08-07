# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 19:42:40 2014

@author: ricardomayerb
"""

import sympy
alpha, beta, delta, nu, mua = sympy.symbols('alpha, beta, delta, nu, mua')
Css, Kss, Lss, Yss, Iss = sympy.symbols('Css, Kss, Lss, Yss, Iss')
ass, zetaass, zetaa_stationary = sympy.symbols(
    'ass, zetaass, zetaa_stationary')

emua = sympy.exp(mua)


theta = nu*(1-alpha)/(1-nu)

B1 = (emua + beta*(delta-1))/(alpha*beta)
B2 = emua + delta - 1
B3 = B1 - B2
B4 = theta*B1/B3
B5 = B4/(1+B4)
B1powalpha = B1**(1/(1-alpha))
B6 = B5*emua/B1powalpha

Kss_sol_sym = B6
Lss_sol_sym = B5
Css_sol_sym = B6*B3
Iss_sol_sym = B6*B2
Yss_sol_sym = B6*B1
zss_sol_sym = 0
ass_sol_sym = mua

# this should be an issue


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

tech_dict = {'alpha': alpha_value, 'delta': delta_value}
prefs_dict = {'beta': beta_value, 'nu': nu_value}
prod_dict = {'rhoa': rhoa_value, 'mua': mua_value_true_Toy,
             'zetaa_stationary': 0}
tech_prefs_prod_dict = {}
tech_prefs_prod_dict.update(tech_dict)
tech_prefs_prod_dict.update(prefs_dict)
tech_prefs_prod_dict.update(prod_dict)

foo = [x.subs(tech_prefs_prod_dict) for x in [B1, B2, B3, B4, B5, B6]]
# print foo

ss_symbolic_solutions_dict = {Css: Css_sol_sym, Iss: Iss_sol_sym,
                              Kss: Kss_sol_sym,
                              Lss: Lss_sol_sym, Yss: Yss_sol_sym,
                              ass: mua, zetaass: zetaa_stationary}


ss_sol_values_from_symb_list = [x.subs(tech_prefs_prod_dict) for x in
                                ss_symbolic_solutions_dict.values()]

# print ss_sol_values_from_symb_list
ss_sol_values_from_symb_dict = dict(
    zip(ss_symbolic_solutions_dict.keys(), ss_sol_values_from_symb_list))

print ss_sol_values_from_symb_dict
