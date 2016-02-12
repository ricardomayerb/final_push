import fipiro
import fipiro.utils as u
import sympy
# from sympy.utilities.lambdify import lambdify
# from math import exp
import numpy as np
reload(fipiro.utils)

non_ex_block_index = (1, 2, 3, 4)
ex_block_index = (0)
# z_block_index = (5, 6)
z_block_index = (5, 6, 7)
uhlig_block_indices = {'expectational_block': ex_block_index,
                       'non_expectational_block': non_ex_block_index,
                       'z_block': z_block_index}
u_x_names = ['K']
u_y_names = ['C', 'I', 'N', 'R', 'Y']
u_z_names = ['lnofZ']

x_names = u_x_names + u_y_names + u_z_names

x_dates = ['tm1', 't', 'tp1']
w_names = ['epsilon']
w_dates = ['t', 'tp1']

param_names = ['A', 'beta', 'delta', 'psi',
               'rho', 'eta', 'sigma']
all_names = {
    'x_names': x_names, 'w_names': w_names, 'param_names': param_names}

xxsswp_sym_d = u.make_x_w_param_sym_dicts(x_names, w_names, param_names)

N_bar = 1.0/3  # Steady state employment is a third of total time endowment
Z_bar = 1  # Normalization
rho_value = .36  # Capital share
delta_value = 0.025  # Depreciation rate for capital
R_bar = 1.01  # One percent real interest per quarter
beta_value = 1/R_bar
eta_value = 1.0  # constant of relative risk aversion = 1/(coeff. of
# intertemporal substitution)
psi_value = 0.95  # autocorrelation of technology shock
sigma_eps = .712  # Standard deviation of technology shock.  Units: Percent.
YK_bar = (R_bar + delta_value - 1)/rho_value
K_bar = (YK_bar / Z_bar)**(1.0/(rho_value-1)) * N_bar
I_bar = delta_value * K_bar
Y_bar = YK_bar * K_bar
C_bar = Y_bar - delta_value*K_bar
A_value = C_bar**(-eta_value) * (1 - rho_value) * Y_bar/N_bar

x_s_d, x_in_ss_sym_d, w_s_d, param_sym_d = xxsswp_sym_d
xss_ini_dict = {x_in_ss_sym_d['Y_ss']: 1.0,
                x_in_ss_sym_d['K_ss']: 10.0,
                x_in_ss_sym_d['lnofZ_ss']: 0.001,
                x_in_ss_sym_d['I_ss']: 0.3,
                x_in_ss_sym_d['R_ss']: 1.0,
                x_in_ss_sym_d['C_ss']: 0.6,
                x_in_ss_sym_d['N_ss']: 0.4}

xxsswp_sym_d = u.make_x_w_param_sym_dicts(x_names, w_names, param_names)
x_s_d, x_in_ss_sym_d, w_s_d, param_sym_d = xxsswp_sym_d

for k, v in x_s_d.iteritems():
    exec(k + '= v')

for k, v in w_s_d.iteritems():
    exec(k + '= v')

for k, v in param_sym_d.iteritems():
    exec(k + '= v')


def g_hansen_1985_div_by_uligh_tk():
    """equations"""
    eq1 = beta * ((Ct/Ctp1)**eta * Rtp1) - 1
    eq2 = Ct + It - Yt
    eq3 = It + (1-delta) * Ktm1 - Kt
    eq4 = sympy.exp(lnofZt) * Ktm1**rho * Nt**(1-rho) - Yt
    eq5 = (1/(Ct**eta)) * (1-rho) * Yt/Nt - A
    eq6 = rho * Yt/Ktm1 + 1 - delta - Rt
    eq7 = (1-psi)*sympy.log(Z_bar) + psi*lnofZtm1 + sigma*epsilont - lnofZt

    return [eq1, eq2, eq3, eq4, eq5, eq6, eq7]

eqs_div_toolkit = g_hansen_1985_div_by_uligh_tk()

u_x_devnames = ['devss' + x for x in u_x_names]
u_y_devnames = ['devss' + y for y in u_y_names]
u_z_devnames = ['devss' + z for z in u_z_names]

x_devnames = ['devss' + x for x in x_names]


pref_tech_names_to_values = {'rho': rho_value, 'beta': beta_value,
                             'delta': delta_value, 'eta': eta_value,
                             'sigma': sigma_eps, 'A': A_value,
                             'psi': psi_value}

all_param_values_dict = {}
all_param_values_dict.update(pref_tech_names_to_values)

# bmodel = u.ModelBase(glist_cikly_az_info, x_names=x_names,
#                      w_names=w_names, param_names=param_names,
#                      par_to_values_dict=all_param_values_dict)


non_ex_block_index = (1, 2, 3, 4, 5)
ex_block_index = (0)
z_block_index = (6)
uhlig_block_indices = {'expectational_block': ex_block_index,
                       'non_expectational_block': non_ex_block_index,
                       'z_block': z_block_index}

umodel = u.UhligModel(eqs_div_toolkit,
                      x_names=x_names, w_names=w_names,
                      param_names=param_names,
                      block_indices=uhlig_block_indices,
                      u_x_names=u_x_names,
                      u_y_names=u_y_names,
                      u_z_names=u_z_names,
                      par_to_values_dict=all_param_values_dict)

#
# umodel = u.UhligModel(eqs_div_toolkit,
#                       x_names=x_names, w_names=w_names,
#                       param_names=param_names,
#                       block_indices=uhlig_block_indices,
#                       u_x_names=u_x_names,
#                       u_y_names=u_y_names,
#                       u_z_names=u_z_names,
#                       par_to_values_dict=all_param_values_dict,
#                       vars_initvalues_dict=xss_ini_dict)



# % VERSION 2.0, MARCH 1997, COPYRIGHT H. UHLIG.
# % EXAMPL1.M calculates through Hansens benchmark real business
# % cycle model in H. Uhlig, "A toolkit for solving nonlinear dynamic stochastic models easily".
# % First, parameters are set and the steady state is calculated. Next, the matrices are
# % declared.  In the last line, the model is solved and analyzed by calling DO_IT.M
#
# % Copyright: H. Uhlig.  Feel free to copy, modify and use at your own risk.
# % However, you are not allowed to sell this software or otherwise impinge
# % on its free distribution.
#
# disp('EXAMPLE 1: Hansen benchmark real business cycle model,');
# disp('           see Hansen, G., "Indivisible Labor and the Business Cycle,"');
# disp('           Journal of Monetary Economics, 16 (1985), 281-308.');
#
# disp('Hit any key when ready...');
# pause;
#
# % Setting parameters:
#
# N_bar     = 1.0/3;  % Steady state employment is a third of total time endowment
# Z_bar     = 1; % Normalization
# rho       = .36; % Capital share
# delta     = .025; % Depreciation rate for capital
# R_bar     = 1.01; % One percent real interest per quarter
# eta       =  1.0; % constant of relative risk aversion = 1/(coeff. of intertemporal substitution)
# psi       = .95; % autocorrelation of technology shock
# sigma_eps = .712; % Standard deviation of technology shock.  Units: Percent.
#
# % Calculating the steady state:
#
# betta   = 1.0/R_bar;  % Discount factor beta
# YK_bar  = (R_bar + delta - 1)/rho;  % = Y_bar / K_bar
# K_bar   = (YK_bar / Z_bar)^(1.0/(rho-1)) * N_bar;
# I_bar   = delta * K_bar;
# Y_bar   = YK_bar * K_bar;
# C_bar   = Y_bar - delta*K_bar;
# A       =  C_bar^(-eta) * (1 - rho) * Y_bar/N_bar; % Parameter in utility function
#
# % Declaring the matrices.
#
#
# VARNAMES = ['capital    ',
#             'consumption',
#             'output     ',
#             'labor      ',
#             'interest   ',
#             'investment ',
#             'technology '];
#
# % Translating into coefficient matrices.
# % The equations are, conveniently ordered:
# % 1) 0 = - I i(t) - C c(t) + Y y(t)
# % 2) 0 = I i(t) - K k(t) + (1-delta) K k(t-1)
# % 3) 0 = rho k(t-1) - y(t) + (1-rho) n(t) + z(t)
# % 4) 0 = -eta c(t) + y(t) - n(t)
# % 5) 0 = - rho Y/K k(t-1) + rho Y/K y(t) - R r(t)
# % 6) 0 = E_t [ - eta c(t+1) + r(t+1) + eta c(t) ]
# % 7) z(t+1) = psi z(t) + epsilon(t+1)
# % CHECK: 7 equations, 7 variables.
# %
# % Endogenous state variables "x(t)": k(t)
# % Endogenous other variables "y(t)": c(t), y(t), n(t), r(t), i(t)
# % Exogenous state variables  "z(t)": z(t).
# % Switch to that notation.  Find matrices for format
# % 0 = AA x(t) + BB x(t-1) + CC y(t) + DD z(t)
# % 0 = E_t [ FF x(t+1) + GG x(t) + HH x(t-1) + JJ y(t+1) + KK y(t) + LL z(t+1) + MM z(t)]
# % z(t+1) = NN z(t) + epsilon(t+1) with E_t [ epsilon(t+1) ] = 0,
#
# % for k(t):
# AA = [ 0
#        - K_bar
#        0
#        0
#        0 ];
#
# % for k(t-1):
# BB = [ 0
#        (1-delta)*K_bar
#        rho
#        0
#        - rho * YK_bar ];
#
# %Order:   consumption  output      labor     interest  investment
# CC = [    -C_bar,      Y_bar,      0,        0,        -I_bar % Equ. 1)
#           0,           0,          0,        0,        I_bar  % Equ. 2)
#           0,           -1,         1-rho,    0,        0      % Equ. 3)
#           -eta,        1,          -1,       0,        0      % Equ. 4)
#           0,           rho*YK_bar, 0,        - R_bar,  0 ];   % Equ. 5)
#
# DD = [ 0
#        0
#        1
#        0
#        0 ];
#
# FF = [ 0 ];
#
# GG = [ 0 ];
#
# HH = [ 0 ];
#
# JJ = [ -eta,  0,  0,  1,  0];
#
# KK = [ eta,   0,  0,  0,  0];
#
# LL = [ 0 ];
#
# MM = [ 0 ];
#
# NN = [psi];
#
# Sigma = [ sigma_eps^2  ];
#
# % Setting the options:
#
# [l_equ,m_states] = size(AA);
# [l_equ,n_endog ] = size(CC);
# [l_equ,k_exog  ] = size(DD);
#
#
# PERIOD     = 4;  % number of periods per year, i.e. 12 for monthly, 4 for quarterly
# GNP_INDEX  = 3; % Index of output among the variables selected for HP filter
# IMP_SELECT = [1:7];
#    %  a vector containing the indices of the variables to be plotted
# DO_SIMUL   = 1; % Calculates simulations
# SIM_LENGTH = 150;
# DO_MOMENTS = 1; % Calculates moments based on frequency-domain methods
# HP_SELECT  = 1:(m_states+n_endog+k_exog); % Selecting the variables for the HP Filter calcs.
# % DO_COLOR_PRINT = 1;
#
#
# % Starting the calculations:
#
# do_it;
