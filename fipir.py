# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 11:12:37 2014

@author: ricardomayerb
"""

import sympy
import scipy.linalg
import numpy as np
import matplotlib.pyplot as plt
import riccati
import theano


#if using R's geigen package to compute the ordered QZ decomposition, we 
#need to load this package via rpy2:
#Note: Scipy 0.15 plans to pack its own ordered QZ
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
robjects.r.library("geigen")
rgqz = robjects.r.gqz





class StateSpace:
    x = 0
    def __init__(self, A=[], C=[], D=[], G=[], z0=0):
        self.A = A
        self.C = C
        self.D = D
        self.G = G
        

#**tech_dict, **prefs_dict, **productivity_dict
#
#def __init__(self, statespace, alpha = 0.33, delta = 0.1, gamma=1,
#                 beta = 0.95, rhoa=0.95, nu=0.5, muatrue = 0.03, 
#                 modeltype = 'toy', statespace_symdict = {}, xvarstring='CKL'):
#        self.A = statespace.A
#        self.C = statespace.C
#        self.D = statespace.D
#        self.G = statespace.G
#        self.alpha = alpha
#        self.nu = nu
#        self.mua = muatrue
#        self.beta = beta
#        self.delta = delta
#        self.gamma = gamma
#        self.rhoa = rhoa
#        self.modeltype = modeltype
#        self.statespace_symdict = statespace_symdict
#        self.xvarstring = xvarstring
#        
#        if self.modeltype=='toy':
#            self.modelsymbols_dict = self.get_modelsymbols('toy')
#            
#        else:
#            self.modelsymbols_dict = []
#            
#    

class FullInfoModel:
    x = 0
    def __init__(self, statespace,  alpha = 0.33, delta = 0.1, gamma=1,
                 beta = 0.95, rhoa=0.95, nu=0.5, muatrue = 0.03, 
                 modeltype = 'toy', statespace_symdict = {},
                 ss_x_dict={}, xvarstring='CKL'):
        self.A = statespace.A
        self.C = statespace.C
        self.D = statespace.D
        self.G = statespace.G
        self.alpha = alpha
        self.nu = nu
        self.mua = muatrue
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.rhoa = rhoa
        self.modeltype = modeltype
        self.statespace_symdict = statespace_symdict
        self.xvarstring = xvarstring
        
        if self.modeltype=='toy':
            self.modelsymbols_dict = self.get_modelsymbols('toy')
            
        else:
            self.modelsymbols_dict = []
            
            
        self.q_sym = self.modelsymbols_dict['q']

        self.at_q = self.modelsymbols_dict['at_q']
        self.atm1_q = self.modelsymbols_dict['atm1_q']
        self.atp1_q = self.modelsymbols_dict['atp1_q']        
        self.Ct_q = self.modelsymbols_dict['Ct_q']
        self.Ctm1_q = self.modelsymbols_dict['Ctm1_q']
        self.Ctp1_q = self.modelsymbols_dict['Ctp1_q']
        self.It_q = self.modelsymbols_dict['It_q']
        self.Itm1_q = self.modelsymbols_dict['Itm1_q']
        self.Itp1_q = self.modelsymbols_dict['Itp1_q']
        self.Kt_q = self.modelsymbols_dict['Kt_q']
        self.Ktm1_q = self.modelsymbols_dict['Ktm1_q']
        self.Ktp1_q = self.modelsymbols_dict['Ktp1_q']
        self.Lt_q = self.modelsymbols_dict['Lt_q']
        self.Ltm1_q = self.modelsymbols_dict['Ltm1_q']
        self.Ltp1_q = self.modelsymbols_dict['Ltp1_q']
        self.Yt_q = self.modelsymbols_dict['Yt_q']
        self.Ytm1_q = self.modelsymbols_dict['Ytm1_q']
        self.Ytp1_q = self.modelsymbols_dict['Ytp1_q']
        self.zetaat_q = self.modelsymbols_dict['zetaat_q']
        self.zetaatm1_q = self.modelsymbols_dict['zetaatm1_q']
        self.zetaatp1_q = self.modelsymbols_dict['zetaatp1_q']
        
        self.a0t = self.modelsymbols_dict['a0t']
        self.a0tm1 = self.modelsymbols_dict['a0tm1']
        self.a0tp1 = self.modelsymbols_dict['a0tp1']        
        self.C0t = self.modelsymbols_dict['C0t']
        self.C0tm1 = self.modelsymbols_dict['C0tm1']
        self.C0tp1 = self.modelsymbols_dict['C0tp1']
        self.I0t = self.modelsymbols_dict['I0t']
        self.I0tm1 = self.modelsymbols_dict['I0tm1']
        self.I0tp1 = self.modelsymbols_dict['I0tp1']
        self.K0t = self.modelsymbols_dict['K0t']
        self.K0tm1 = self.modelsymbols_dict['K0tm1']
        self.K0tp1 = self.modelsymbols_dict['K0tp1']
        self.L0t = self.modelsymbols_dict['L0t']
        self.L0tm1 = self.modelsymbols_dict['L0tm1']
        self.L0tp1 = self.modelsymbols_dict['L0tp1']
        self.Y0t = self.modelsymbols_dict['Y0t']
        self.Y0tm1 = self.modelsymbols_dict['Y0tm1']
        self.Y0tp1 = self.modelsymbols_dict['Y0tp1']
        self.zetaa0t = self.modelsymbols_dict['zetaa0t']
        self.zetaa0tm1 = self.modelsymbols_dict['zetaa0tm1']
        self.zetaa0tp1 = self.modelsymbols_dict['zetaa0tp1']
        
        self.a1t = self.modelsymbols_dict['a1t']
        self.a1tm1 = self.modelsymbols_dict['a1tm1']
        self.a1tp1 = self.modelsymbols_dict['a1tp1']        
        self.C1t = self.modelsymbols_dict['C1t']
        self.C1tm1 = self.modelsymbols_dict['C1tm1']
        self.C1tp1 = self.modelsymbols_dict['C1tp1']
        self.I1t = self.modelsymbols_dict['I1t']
        self.I1tm1 = self.modelsymbols_dict['I1tm1']
        self.I1tp1 = self.modelsymbols_dict['I1tp1']
        self.K1t = self.modelsymbols_dict['K1t']
        self.K1tm1 = self.modelsymbols_dict['K1tm1']
        self.K1tp1 = self.modelsymbols_dict['K1tp1']
        self.L1t = self.modelsymbols_dict['L1t']
        self.L1tm1 = self.modelsymbols_dict['L1tm1']
        self.L1tp1 = self.modelsymbols_dict['L1tp1']
        self.Y1t = self.modelsymbols_dict['Y1t']
        self.Y1tm1 = self.modelsymbols_dict['Y1tm1']
        self.Y1tp1 = self.modelsymbols_dict['Y1tp1']
        self.zetaa1t = self.modelsymbols_dict['zetaa1t']
        self.zetaa1tm1 = self.modelsymbols_dict['zetaa1tm1']
        self.zetaa1tp1 = self.modelsymbols_dict['zetaa1tp1']
                
        self.a2t = self.modelsymbols_dict['a2t']
        self.a2tm1 = self.modelsymbols_dict['a2tm1']
        self.a2tp1 = self.modelsymbols_dict['a2tp1']        
        self.C2t = self.modelsymbols_dict['C2t']
        self.C2tm1 = self.modelsymbols_dict['C2tm1']
        self.C2tp1 = self.modelsymbols_dict['C2tp1']
        self.I2t = self.modelsymbols_dict['I2t']
        self.I2tm1 = self.modelsymbols_dict['I2tm1']
        self.I2tp1 = self.modelsymbols_dict['I2tp1']
        self.K2t = self.modelsymbols_dict['K2t']
        self.K2tm1 = self.modelsymbols_dict['K2tm1']
        self.K2tp1 = self.modelsymbols_dict['K2tp1']
        self.L2t = self.modelsymbols_dict['L2t']
        self.L2tm1 = self.modelsymbols_dict['L2tm1']
        self.L2tp1 = self.modelsymbols_dict['L2tp1']
        self.Y2t = self.modelsymbols_dict['Y2t']
        self.Y2tm1 = self.modelsymbols_dict['Y2tm1']
        self.Y2tp1 = self.modelsymbols_dict['Y2tp1']
        self.zetaa2t = self.modelsymbols_dict['zetaa2t']
        self.zetaa2tm1 = self.modelsymbols_dict['zetaa2tm1']
        self.zetaa2tp1 = self.modelsymbols_dict['zetaa2tp1']
        
        
        self.Ct_sym = self.modelsymbols_dict['Ct']
        self.Ctm1_sym = self.modelsymbols_dict['Ctm1']
        self.Ctp1_sym = self.modelsymbols_dict['Ctp1']
        
        self.Lt_sym = self.modelsymbols_dict['Lt']
        self.Ltm1_sym = self.modelsymbols_dict['Ltm1']
        self.Ltp1_sym = self.modelsymbols_dict['Ltp1']
        
        self.Ytm1_sym = self.modelsymbols_dict['Ytm1']
        self.Yt_sym = self.modelsymbols_dict['Yt']
        self.Ytp1_sym = self.modelsymbols_dict['Ytp1']
        
        self.Itm1_sym = self.modelsymbols_dict['Itm1']
        self.It_sym = self.modelsymbols_dict['It']
        self.Itp1_sym = self.modelsymbols_dict['Itp1']
        
        self.Kstate_sym = self.modelsymbols_dict['Kstate']
        self.Kpolicy_sym = self.modelsymbols_dict['Kpolicy']
        self.Ktp2_sym = self.modelsymbols_dict['Ktp2']
        
        self.Css_sym = self.modelsymbols_dict['Css']
        self.Kss_sym = self.modelsymbols_dict['Kss']
        self.Lss_sym = self.modelsymbols_dict['Lss']
        self.Iss_sym = self.modelsymbols_dict['Iss']
        self.Yss_sym = self.modelsymbols_dict['Yss']
        
        self.muass_sym = self.modelsymbols_dict['muass']    
        self.zetaass_sym = self.modelsymbols_dict['zetaass']    
       
       
        self.A_sym = self.modelsymbols_dict['A']
        self.C_sym = self.modelsymbols_dict['C']
        self.D_sym = self.modelsymbols_dict['D']
        self.G_sym = self.modelsymbols_dict['G']
        
        self.rhoa_sym = self.modelsymbols_dict['rhoa']
        self.gamma_sym = self.modelsymbols_dict['gamma']
        self.alpha_sym = self.modelsymbols_dict['alpha']
        self.beta_sym = self.modelsymbols_dict['beta']
        self.delta_sym = self.modelsymbols_dict['delta']
        self.nu_sym = self.modelsymbols_dict['nu']
        
        self.mua_sym = self.modelsymbols_dict['mua']
        self.zetaat_sym = self.modelsymbols_dict['zetaat']
        self.zetaatm1_sym = self.modelsymbols_dict['zetaatm1']
        self.zetaatp1_sym = self.modelsymbols_dict['zetaatp1']
        self.wat_sym = self.modelsymbols_dict['wat']
        self.watm1_sym = self.modelsymbols_dict['watm1']
        self.watp1_sym = self.modelsymbols_dict['watp1']
        self.wzetaat_sym = self.modelsymbols_dict['wzetaat']
        self.wzetaatm1_sym = self.modelsymbols_dict['wzetaatm1']
        self.wzetaatp1_sym = self.modelsymbols_dict['wzetaatp1']

        self.muv_sym = self.modelsymbols_dict['muv']
        self.zetavt_sym = self.modelsymbols_dict['zetavt']
        self.zetavtm1_sym = self.modelsymbols_dict['zetavtm1']
        self.zetavtp1_sym = self.modelsymbols_dict['zetavtp1']
        self.wvt_sym = self.modelsymbols_dict['wvt']
        self.wvtm1_sym = self.modelsymbols_dict['wvtm1']
        self.wvtp1_sym = self.modelsymbols_dict['wvtp1']
        
        self.at_sym   = self.modelsymbols_dict['at']
        self.atp1_sym   = self.modelsymbols_dict['atp1']


        self.zt_sym = self.modelsymbols_dict['zt']
        self.ztm1_sym = self.modelsymbols_dict['ztm1']
        self.ztp1_sym = self.modelsymbols_dict['ztp1']
        self.zetat_sym = self.modelsymbols_dict['zetat']
        self.zetatm1_sym = self.modelsymbols_dict['zetatm1']
        self.zetatp1_sym = self.modelsymbols_dict['zetatp1']
        
        self.wt_sym = self.modelsymbols_dict['wt']
        self.wtm1_sym = self.modelsymbols_dict['wtm1']
        self.wtp1_sym = self.modelsymbols_dict['wtp1']
        
        
        
        self.dictionary_sym2numval_params = {
            self.alpha_sym:self.alpha, self.beta_sym:self.beta,
            self.delta_sym:self.delta, self.nu_sym:self.nu,
            self.rhoa_sym:self.rhoa, self.mua_sym:self.mua }
            
        self.ss_solutions_dict = ss_x_dict
        
        if self.xvarstring=='CKL':
            self.xvar_tm1_sym = [self.Ctm1_sym, self.Kstate_sym,
                                 self.Ltm1_sym, self.zetaatm1_sym]
            self.xvar_t_sym = [self.Ct_sym, self.Kpolicy_sym,
                               self.Lt_sym, self.zetaat_sym]
            self.xvar_tp1_sym = [self.Ctp1_sym, self.Ktp2_sym,
                                 self.Ltp1_sym, self.zetaatp1_sym]
            self.xvar_atss_sym = [self.Css_sym, self.Kss_sym, self.Lss_sym,
                                  self.zetaass_sym]
            
        elif self.xvarstring=='CKLIY':
            self.xvar_tm1_sym = [self.Ctm1_sym, self.Kstate_sym, self.Ltm1_sym,
                                 self.Itm1_sym, self.Ytm1_sym, self.zetaatm1_sym]
            self.xvar_t_sym = [self.Ct_sym, self.Kpolicy_sym, self.Lt_sym,
                               self.It_sym, self.Yt_sym, self.zetaat_sym]
            self.xvar_tp1_sym = [self.Ctp1_sym, self.Ktp2_sym, self.Ltp1_sym,
                                 self.Itp1_sym, self.Ytp1_sym, self.zetaatp1_sym]

            self.xvar_atss_sym = [self.Css_sym, self.Kss_sym, self.Lss_sym,
                                  self.Iss_sym, self.Yss_sym, self.zetaass_sym]
                                  
             
        self.wvar_t_sym = [self.wat_sym, self.wzetaat_sym]

        self.wvar_tp1_sym = [self.watp1_sym, self.wzetaatp1_sym] 
                    
#        deriv_vars_w_t = [self.wat_sym, self.wzetaat_sym]
#
#        deriv_vars_w_tp1 = [self.watp1_sym, self.wzetaatp1_sym]
#                                  
                                  
        self.dictionary_with_steady_state_versions = {
            self.Ct_sym : self.Css_sym, self.Lt_sym : self.Lss_sym, 
            self.Kstate_sym : self.Kss_sym, self.Kpolicy_sym : self.Kss_sym,
            self.Ctp1_sym : self.Css_sym, self.Ltp1_sym : self.Lss_sym, 
            self.Ytp1_sym : self.Yss_sym, self.Itp1_sym : self.Iss_sym,
            self.Ytm1_sym : self.Yss_sym, self.Itm1_sym : self.Iss_sym,
            self.Yt_sym : self.Yss_sym, self.It_sym : self.Iss_sym,
            self.mua_sym : self.muass_sym, self.zetaatm1_sym:0,
            self.zetaat_sym:0, self.zetaatp1_sym : 0, self.wat_sym : 0, 
            self.watp1_sym : 0, self.wzetaat_sym:0, self.wzetaatp1_sym:0,
            self.at_sym:self.mua,self.atp1_sym:self.mua} 
                
        
        
    def get_modelsymbols(self, modtype='toy'):
        
        # variables with NTM means not used in toy model
        # but I defined to be able to take derivs w.r.t them, make things 
        # easier to write
        
        
        q_sym = sympy.Symbol('q') 
        
#        at_q = sympy.Symbol('at_q')
#        atm1_q = sympy.Symbol('atm1_q')
#        atp1_q = sympy.Symbol('atp1_q')        
#        Ct_q = sympy.Symbol('Ct_q')
#        Ctm1_q = sympy.Symbol('Ctm1_q')
#        Ctp1_q = sympy.Symbol('Ctp1_q')
#        It_q = sympy.Symbol('It_q')
#        Itm1_q = sympy.Symbol('Itm1_q')
#        Itp1_q = sympy.Symbol('Itp1_q')
#        Kt_q = sympy.Symbol('Kt_q')
#        Ktm1_q = sympy.Symbol('Ktm1_q')
#        Ktp1_q = sympy.Symbol('Ktp1_q')
#        Lt_q = sympy.Symbol('Lt_q')
#        Ltm1_q = sympy.Symbol('Ltm1_q')
#        Ltp1_q = sympy.Symbol('Ltp1_q')
#        Yt_q = sympy.Symbol('Yt_q')
#        Ytm1_q = sympy.Symbol('Ytm1_q')
#        Ytp1_q = sympy.Symbol('Ytp1_q')
#        zetaat_q = sympy.Symbol('zetaat_q')
#        zetaatm1_q = sympy.Symbol('zetaatm1_q')
#        zetaatp1_q = sympy.Symbol('zetaatp1_q')
        
        at_q = sympy.Function('at')(q_sym) 
        atm1_q = sympy.Function('atm1')(q_sym) 
        atp1_q = sympy.Function('atp1')(q_sym)         
        Ct_q = sympy.Function('Ct')(q_sym) 
        Ctm1_q = sympy.Function('Ctm1')(q_sym) 
        Ctp1_q = sympy.Function('Ctp1')(q_sym) 
        It_q = sympy.Function('It')(q_sym) 
        Itm1_q = sympy.Function('Itm1')(q_sym) 
        Itp1_q = sympy.Function('Itp1')(q_sym) 
        Kt_q = sympy.Function('Kt')(q_sym) 
        Ktm1_q = sympy.Function('Ktm1')(q_sym) 
        Ktp1_q = sympy.Function('Ktp1')(q_sym) 
        Lt_q = sympy.Function('Lt')(q_sym) 
        Ltm1_q = sympy.Function('Ltm1')(q_sym) 
        Ltp1_q = sympy.Function('Ltp1')(q_sym) 
        Yt_q = sympy.Function('Yt')(q_sym) 
        Ytm1_q = sympy.Function('Ytm1')(q_sym) 
        Ytp1_q = sympy.Function('Ytp1')(q_sym) 
        zetaat_q = sympy.Function('zetaat')(q_sym) 
        zetaatm1_q = sympy.Function('zetaatm1')(q_sym) 
        zetaatp1_q = sympy.Function('zetaatp1')(q_sym) 
        
        a0t = sympy.Symbol('a0t')
        a0tm1 = sympy.Symbol('a0tm1')
        a0tp1 = sympy.Symbol('a0tp1')        
        C0t = sympy.Symbol('C0t')
        C0tm1 = sympy.Symbol('C0tm1')
        C0tp1 = sympy.Symbol('C0tp1')
        I0t = sympy.Symbol('I0t')
        I0tm1 = sympy.Symbol('I0tm1')
        I0tp1 = sympy.Symbol('I0tp1')
        K0t = sympy.Symbol('K0t')
        K0tm1 = sympy.Symbol('K0tm1')
        K0tp1 = sympy.Symbol('K0tp1')
        L0t = sympy.Symbol('L0t')
        L0tm1 = sympy.Symbol('L0tm1')
        L0tp1 = sympy.Symbol('L0tp1')
        Y0t = sympy.Symbol('Y0t')
        Y0tm1 = sympy.Symbol('Y0tm1')
        Y0tp1 = sympy.Symbol('Y0tp1')
        zetaa0t = sympy.Symbol('zetaa0t')
        zetaa0tm1 = sympy.Symbol('zetaa0tm1')
        zetaa0tp1 = sympy.Symbol('zetaa0tp1')
        
        a1t = sympy.Symbol('a1t')
        a1tm1 = sympy.Symbol('a1tm1')
        a1tp1 = sympy.Symbol('a1tp1')        
        C1t = sympy.Symbol('C1t')
        C1tm1 = sympy.Symbol('C1tm1')
        C1tp1 = sympy.Symbol('C1tp1')
        I1t = sympy.Symbol('I1t')
        I1tm1 = sympy.Symbol('I1tm1')
        I1tp1 = sympy.Symbol('I1tp1')
        K1t = sympy.Symbol('K1t')
        K1tm1 = sympy.Symbol('K1tm1')
        K1tp1 = sympy.Symbol('K1tp1')
        L1t = sympy.Symbol('L1t')
        L1tm1 = sympy.Symbol('L1tm1')
        L1tp1 = sympy.Symbol('L1tp1')
        Y1t = sympy.Symbol('Y1t')
        Y1tm1 = sympy.Symbol('Y1tm1')
        Y1tp1 = sympy.Symbol('Y1tp1')
        zetaa1t = sympy.Symbol('zetaa1t')
        zetaa1tm1 = sympy.Symbol('zetaa1tm1')
        zetaa1tp1 = sympy.Symbol('zetaa1tp1')
                
        a2t = sympy.Symbol('a2t')
        a2tm1 = sympy.Symbol('a2tm1')
        a2tp1 = sympy.Symbol('a2tp1')        
        C2t = sympy.Symbol('C2t')
        C2tm1 = sympy.Symbol('C2tm1')
        C2tp1 = sympy.Symbol('C2tp1')
        I2t = sympy.Symbol('I2t')
        I2tm1 = sympy.Symbol('I2tm1')
        I2tp1 = sympy.Symbol('I2tp1')
        K2t = sympy.Symbol('K2t')
        K2tm1 = sympy.Symbol('K2tm1')
        K2tp1 = sympy.Symbol('K2tp1')
        L2t = sympy.Symbol('L2t')
        L2tm1 = sympy.Symbol('L2tm1')
        L2tp1 = sympy.Symbol('L2tp1')
        Y2t = sympy.Symbol('Y2t')
        Y2tm1 = sympy.Symbol('Y2tm1')
        Y2tp1 = sympy.Symbol('Y2tp1')
        zetaa2t = sympy.Symbol('zetaa2t')
        zetaa2tm1 = sympy.Symbol('zetaa2tm1')
        zetaa2tp1 = sympy.Symbol('zetaa2tp1')
       
        
        Ctm1_sym = sympy.Symbol('C_tm1', positive=True) 
        Ltm1_sym = sympy.Symbol('L_tm1', positive=True) 
        Kstate_sym = sympy.Symbol('K_t', positive=True)
        Ytm1_sym = sympy.Symbol('Y_tm1', positive=True)
        Itm1_sym = sympy.Symbol('I_tm1')
        
        Yt_sym = sympy.Symbol('Y_t', positive=True)
        It_sym = sympy.Symbol('I_t')
        Ct_sym = sympy.Symbol('C_t', positive=True)
        Lt_sym = sympy.Symbol('L_t', positive=True)
        Kpolicy_sym = sympy.Symbol('K_tp1', positive=True)  
 
        Ytp1_sym = sympy.Symbol('Y_tp1', positive=True)
        Ltp1_sym = sympy.Symbol('L_tp1', positive=True)
        Itp1_sym = sympy.Symbol('I_tp1')
        Ctp1_sym = sympy.Symbol('C_tp1', positive=True) 
        Ktp2_sym = sympy.Symbol('K_tp2', positive=True)
        
        Css_sym = sympy.Symbol('Css', positive=True)
        Lss_sym = sympy.Symbol('Lss', positive=True)
        Kss_sym = sympy.Symbol('Kss', positive=True)
        Iss_sym = sympy.Symbol('Iss', positive=True)
        Yss_sym = sympy.Symbol('Yss', positive=True)
        muass_sym = sympy.Symbol('muass', positive=True)
        zetaass_sym = sympy.Symbol('zetaass', positive=True)
                
        alpha_sym = sympy.Symbol('alpha', positive=True)
        beta_sym = sympy.Symbol('beta', positive=True)
        nu_sym = sympy.Symbol('nu', positive=True)
        delta_sym = sympy.Symbol('delta', positive=True)    
        
        at_sym = sympy.Symbol('a_t', real=True) 
        atp1_sym = sympy.Symbol('a_tp1', real=True) 
        zetaat_sym = sympy.Symbol('\zeta_at', real=True) 
        zetaatp1_sym = sympy.Symbol('\zeta_atp1', real=True) 
        zetaatm1_sym = sympy.Symbol('\zeta_atm1', real=True) 
        wat_sym = sympy.Symbol('w_at', real=True) 
        watm1_sym = sympy.Symbol('w_atm1', real=True) 
        watp1_sym = sympy.Symbol('w_atp1', real=True) 
        wzetaat_sym = sympy.Symbol('w_{\zeta_a,t}', real=True)
        wzetaatm1_sym = sympy.Symbol('w_{\zeta_a,t-1}', real=True)
        wzetaatp1_sym = sympy.Symbol('w_{\zeta_a,t+1}', real=True)
    
        mua_sym = sympy.Symbol('mu_a', positive=True)
        rhoa_sym = sympy.Symbol('rho_a', positive=True) 
        
        Czzetaawa_sym = sympy.Symbol( 'C_{z,\zeta_a,w_a}', positive=True) 
        Czzetaawzetaa_sym  = sympy.Symbol( 'C_{z,\zeta_a,\zeta_a}', positive=True)
        
        Gsawa_sym = sympy.Symbol( 'G_{s,a,w_a}', positive=True)
        Gsawzetaa_sym  = sympy.Symbol( 'G_{s,a,w_{\zeta_a}}', positive=True)
        
        Sigmamuamua_sym = sympy.Symbol('\Sigma_{\mu_a, mu_a}', positive=True)  
        Sigmamuazetaa_sym = sympy.Symbol('\Sigma_{\mu_{a},\zeta_{a}}', positive=True)  
        Sigmazetaazetaa_sym = sympy.Symbol('\Sigma_{\zeta_{a},\zeta_{a}}', positive=True)
        
        if self.statespace_symdict.has_key('A'):
            Az_sym = self.statespace_symdict['A']
        else:
            Az_sym = sympy.Matrix([[1.0, 0],[0, rhoa_sym]])
            
        if self.statespace_symdict.has_key('C'):
            Cz_sym = self.statespace_symdict['C']
        else:
            Cz_sym = sympy.Matrix([[0,0],[Czzetaawa_sym, Czzetaawzetaa_sym]])
        
        if self.statespace_symdict.has_key('D'):
            Ds_sym = self.statespace_symdict['D']
        else:
            Ds_sym = sympy.Matrix([[1, 1]])

        if self.statespace_symdict.has_key('G'):
            Gs_sym = self.statespace_symdict['G']
        else:
            Gs_sym = sympy.Matrix([[Gsawa_sym, Gsawzetaa_sym]])
         
         
        Sigma_sym = sympy.Matrix([[Sigmamuamua_sym, Sigmamuazetaa_sym],[Sigmamuazetaa_sym, Sigmazetaazetaa_sym]])
        
        wt_sym = sympy.Matrix([[wat_sym],[wzetaat_sym]])
        wtm1_sym = sympy.Matrix([[watm1_sym],[wzetaatm1_sym]])
        wtp1_sym = sympy.Matrix([[watp1_sym],[wzetaatp1_sym]])
        zetat_sym = sympy.Matrix([[zetaat_sym]])
        zetatp1_sym = sympy.Matrix([[zetaatp1_sym]])
        zetatm1_sym = sympy.Matrix([[zetaatm1_sym]])
        mu_sym = sympy.Matrix([[mua_sym]])    
        zt_sym = sympy.Matrix([[mua_sym],[zetaat_sym]])
        ztp1_sym = sympy.Matrix([[mua_sym],[zetaatp1_sym]])
        ztm1_sym = sympy.Matrix([[mua_sym],[zetaatm1_sym]])
        
        
        # set symbols absent in the toy model equal to empty lists:
        gamma_sym = []
        vt_sym = []
        vtp1_sym = []
        zetavt_sym = []
        zetavtp1_sym = [] 
        zetavtm1_sym = []
        wvt_sym = []
        wvtm1_sym = []
        wvtp1_sym = []
        wzetavt_sym = []
        wzetavtm1_sym = []
        wzetavtp1_sym = []
        muv_sym = []
            
    
        model_symbols_dict = { 'q':q_sym,
            'a0tm1':a0tm1,'a0t':a0t, 'a0tp1':a0tp1,  
            'C0tm1':C0tm1,'C0t':C0t, 'C0tp1':C0tp1,  
            'I0tm1':I0tm1,'I0t':I0t, 'I0tp1':I0tp1,  
            'K0tm1':K0tm1,'K0t':K0t, 'K0tp1':K0tp1,  
            'L0tm1':L0tm1,'L0t':L0t, 'L0tp1':L0tp1,  
            'Y0tm1':Y0tm1,'Y0t':Y0t, 'Y0tp1':Y0tp1,  
            'zetaa0tm1':zetaa0tm1,'zetaa0t':zetaa0t, 'zetaa0tp1':zetaa0tp1,  
            'a1tm1':a1tm1,'a1t':a1t, 'a1tp1':a1tp1,  
            'C1tm1':C1tm1,'C1t':C1t, 'C1tp1':C1tp1,  
            'I1tm1':I1tm1,'I1t':I1t, 'I1tp1':I1tp1,  
            'K1tm1':K1tm1,'K1t':K1t, 'K1tp1':K1tp1,  
            'L1tm1':L1tm1,'L1t':L1t, 'L1tp1':L1tp1,  
            'Y1tm1':Y1tm1,'Y1t':Y1t, 'Y1tp1':Y1tp1,  
            'zetaa1tm1':zetaa1tm1,'zetaa1t':zetaa1t, 'zetaa1tp1':zetaa1tp1,  
            'a2tm1':a2tm1,'a2t':a2t, 'a2tp1':a2tp1,  
            'C2tm1':C2tm1,'C2t':C2t, 'C2tp1':C2tp1,  
            'I2tm1':I2tm1,'I2t':I2t, 'I2tp1':I2tp1,  
            'K2tm1':K2tm1,'K2t':K2t, 'K2tp1':K2tp1,  
            'L2tm1':L2tm1,'L2t':L2t, 'L2tp1':L2tp1,  
            'Y2tm1':Y2tm1,'Y2t':Y2t, 'Y2tp1':Y2tp1,  
            'zetaa2tm1':zetaa2tm1,'zetaa2t':zetaa2t, 'zetaa2tp1':zetaa2tp1,  
            'atm1_q':atm1_q,'at_q':at_q, 'atp1_q':atp1_q,  
            'Ctm1_q':Ctm1_q,'Ct_q':Ct_q, 'Ctp1_q':Ctp1_q,  
            'Itm1_q':Itm1_q,'It_q':It_q, 'Itp1_q':Itp1_q,  
            'Ktm1_q':Ktm1_q,'Kt_q':Kt_q, 'Ktp1_q':Ktp1_q,  
            'Ltm1_q':Ltm1_q,'Lt_q':Lt_q, 'Ltp1_q':Ltp1_q,  
            'Ytm1_q':Ytm1_q,'Yt_q':Yt_q, 'Ytp1_q':Ytp1_q,  
            'zetaatm1_q':zetaatm1_q,'zetaat_q':zetaat_q, 'zetaatp1_q':zetaatp1_q,  
            'Ct': Ct_sym, 'Ctm1':Ctm1_sym, 'Lt':Lt_sym, 'Ltm1':Ltm1_sym,
            'Yt':Yt_sym, 'It':It_sym, 'Ytm1':Ytm1_sym, 'Itm1':Itm1_sym,
            'Kstate':Kstate_sym, 'Kpolicy':Kpolicy_sym, 'Ltp1':Ltp1_sym,
            'Ctp1':Ctp1_sym, 'Ytp1':Ytp1_sym, 'Itp1':Itp1_sym, 'Ktp2':Ktp2_sym,
            'Css':Css_sym, 'Kss':Kss_sym, 'Lss':Lss_sym, 'Iss':Iss_sym,
            'Yss':Yss_sym,
            'at':at_sym, 'atp1':atp1_sym, 'zetaat':zetaat_sym,
            'zetaatm1':zetaatm1_sym, 'zetaatp1':zetaatp1_sym, 'wat':wat_sym,
            'watm1':watm1_sym, 'watp1':watp1_sym, 'wzetaat':wzetaat_sym,
            'wzetaatm1':wzetaatm1_sym, 'wzetaatp1':wzetaatp1_sym,
            'mua':mua_sym,
            'rhoa':rhoa_sym, 'Czzetaawa':Czzetaawa_sym,
            'Czzetaawa':Czzetaawa_sym, 'Gsawa':Gsawa_sym,
            'Gsawzetaa':Gsawzetaa_sym, 'Sigmamuamua':Sigmamuamua_sym,
            'Sigmamuazetaa':Sigmamuazetaa_sym,
            'Sigmazetaazetaa':Sigmazetaazetaa_sym, 'wt':wt_sym,
            'wtp1':wtp1_sym,
            'zt':zt_sym, 'ztm1':ztm1_sym, 'ztp1':ztp1_sym, 'zetat':zetat_sym,
            'zetatm1':zetatm1_sym, 'zetatp1':zetatp1_sym, 'mu':mu_sym, 
            'A':Az_sym, 'C':Cz_sym, 'D':Ds_sym, 'G':Gs_sym, 'Sigma':Sigma_sym,
            'alpha':alpha_sym, 'beta':beta_sym, 'delta':delta_sym, 'nu':nu_sym,
            'gamma':gamma_sym, 'muv':muv_sym, 'zetavt':zetavt_sym,
            'zetavtm1':zetaatm1_sym, 'zetavtp1':zetavtp1_sym,
            'wvt':wvt_sym,'wvtm1':wvtm1_sym,'wvtp1':wvtp1_sym,
            'wtm1':wtm1_sym, 'muass': muass_sym, 'zetaass': zetaass_sym
            }
                              
        return model_symbols_dict
        
    def gvec_toymodel_full_info_sympy(self, symbolic_elem = 'just_x_w'):
        r"""
        Return a list with the LHS of 
        $g(x_tp1,x_t,x_tm1,w_tp1,w_t)=0$, describing the equilibrium
        conditions of the model and using user-supplied parameter values.
        Only variables in $x$ and $w$ remain as symbols. 
        The main goal is to provide a way to evaluate approximation errors and
        a sympy expression to differentiate w.r.t $x$ and $w$
        """        
        

        if symbolic_elem=='all_but_volatilities':
            alpha, beta = self.alpha_sym, self.beta_sym
            mua = self.mua_sym
            delta, nu = self.delta_sym, self.nu_sym
            A, D = self.A_sym, self.D_sym
            C, G =  self.C, self.G
       
            
        elif symbolic_elem=='just_x_w':
            alpha, beta = self.alpha, self.beta
            mua = self.mua
            delta, nu  = self.delta, self.nu
            A, C, D, G = self.A, self.C, self.D, self.G
            
 
        elif symbolic_elem=='all':
            alpha, beta = self.alpha_sym, self.beta_sym
            delta, nu = self.delta_sym, self.nu_sym
            mua = self.mua_sym
            A, C, D, G = self.A_sym, self.C_sym, self.D_sym, self.G_sym
       
        
        zt_sym, ztm1_sym, zetaat_sym = self.zt_sym, self.ztm1_sym, self.zetaat_sym
        wt_sym, wtp1_sym = self.wt_sym, self.wtp1_sym
        
        at, atp1 = self.at_sym, self.atp1_sym
        
        Ct, Lt, Kt =  self.Ct_sym, self.Lt_sym, self.Kstate_sym 
        It, Yt  =  self.It_sym, self.Yt_sym 
        
        Ctp1, Ltp1, Ktp1 =  self.Ctp1_sym, self.Ltp1_sym, self.Kpolicy_sym 
        Ytp1  =  self.Ytp1_sym 
        
#        at = (D*ztm1_sym + G*wt_sym)[0]
#        atp1 = (D *zt_sym + G*wtp1_sym)[0]
        
#        print '\nD:', D         
#        print '\nztm1_sym:', ztm1_sym         
#        print '\nD*ztm1_sym:', D*ztm1_sym         
#        print '\n(D*ztm1_sym + G*wt_sym):', (D*ztm1_sym + G*wt_sym)         
#        print '\n(D*ztm1_sym + G*wt_sym)[0]:', (D*ztm1_sym + G*wt_sym)[0]
         
        
        at_expr = (D*ztm1_sym + G*wt_sym)[0]
        atp1_expr = (D *zt_sym + G*wtp1_sym)[0]
        
        at_expr = at_expr.subs([(self.mua_sym, mua)])         
        atp1_expr = atp1_expr.subs([(self.mua_sym, mua)])         
        
        It_expr = sympy.exp(at) * Ktp1 - (1-delta)*Kt
        
        
        Yt_expr = sympy.exp((1-alpha)*at)  * Kt**(alpha) * Lt**(1-alpha) 
        Ytp1_expr = sympy.exp((1-alpha)*atp1)  * Ktp1**(alpha) * Ltp1**(1-alpha)
        
        g1_cikly_az = beta*sympy.exp(-at) * (Ct/Ctp1) * \
                (alpha * Ytp1/Ktp1 + (1-delta)) - 1
                
        g2_cikly_az = Yt - It - Ct 
        
        g3_cikly_az = nu*(1-alpha)*Yt/(Ct*(1-nu)) - Lt/(1-Lt)
        
        g4_cikly_az = zetaat_sym - (A*ztm1_sym + C*wt_sym)[1]
        
        g5_cikly_az = at - at_expr
        
        g6_cikly_az = Yt - Yt_expr
        
        g7_cikly_az = It - It_expr 
        
        g1_cikl_az = g1_cikly_az.subs([(Ytp1, Ytp1_expr)])         
        g2_cikl_az = g2_cikly_az.subs([(Yt, Yt_expr)])                                               
        g3_cikl_az = g3_cikly_az.subs([(Yt, Yt_expr)])                     
        g4_cikl_az = g4_cikly_az                     
        g5_cikl_az = g5_cikly_az                     
        g6_cikl_az = g7_cikly_az                     
        
        g1_ckly_az = g1_cikly_az        
        g2_ckly_az = g2_cikly_az.subs([(It, It_expr)])                                               
        g3_ckly_az = g3_cikly_az                     
        g4_ckly_az = g4_cikly_az                     
        g5_ckly_az = g5_cikly_az                     
        g6_ckly_az = g6_cikly_az                     
        
        g1_ckl_az = g1_cikly_az.subs([(Ytp1, Ytp1_expr),(at, at_expr)])         
        g2_ckl_az = g2_cikly_az.subs([(It, It_expr),(Yt, Yt_expr)])                                               
        g3_ckl_az = g3_cikly_az.subs([(Yt, Yt_expr)])                       
        g4_ckl_az = g4_cikly_az                     
        g5_ckl_az = g5_cikly_az                     

        g1_cikly_z = g1_cikly_az.subs([(at, at_expr)])
        g2_cikly_z = g2_cikly_az 
        g3_cikly_z = g3_cikly_az
        g4_cikly_z = g4_cikly_az
        g5_cikly_z = g6_cikly_az.subs([(at, at_expr)])
        g6_cikly_z = g7_cikly_az.subs([(at, at_expr)])
        
        g1_cikl_z = g1_cikly_z.subs([(Ytp1, Ytp1_expr)]).subs([(atp1, atp1_expr)])         
        g2_cikl_z = g2_cikly_z.subs([(Yt, Yt_expr)]).subs([(at, at_expr)])                                               
        g3_cikl_z = g3_cikly_z.subs([(Yt, Yt_expr)]).subs([(at, at_expr)])                     
        g4_cikl_z = g4_cikly_z                     
        g5_cikl_z = g6_cikly_z.subs([(at, at_expr)])                     
        
        g1_ckly_z = g1_cikly_z
        g2_ckly_z = g2_cikly_z.subs([(It, It_expr)]).subs([(at, at_expr)])  
        g3_ckly_z = g3_cikly_z
        g4_ckly_z = g4_cikly_z
        g5_ckly_z = g5_cikly_z
        
        g1_ckl_z = g1_cikl_z      
        g2_ckl_z = g2_cikl_z.subs([(It, It_expr)]).subs([(at, at_expr)])                                               
        g3_ckl_z = g3_cikl_z                     
        g4_ckl_z = g4_cikl_z                     
        
        
        
        glist_cikly_az = [g1_cikly_az, g2_cikly_az, g3_cikly_az, g4_cikly_az,
                          g5_cikly_az, g6_cikly_az, g7_cikly_az]
                          
        glist_ckly_az = [g1_ckly_az, g2_ckly_az, g3_ckly_az, g4_ckly_az,
                          g5_ckly_az, g6_ckly_az]

        glist_cikl_az = [g1_cikl_az, g2_cikl_az, g3_cikl_az, g4_cikl_az,
                          g5_cikl_az, g6_cikl_az]
        
        glist_ckl_az = [g1_ckl_az, g2_ckl_az, g3_ckl_az, g4_ckl_az,
                          g5_ckl_az]
                          
        glist_cikly_z = [g1_cikly_z, g2_cikly_z, g3_cikly_z, g4_cikly_z,
                          g5_cikly_z, g6_cikly_z]

        glist_cikl_z = [g1_cikl_z, g2_cikl_z, g3_cikl_z, g4_cikl_z,
                          g5_cikl_z]
 
        glist_ckly_z = [g1_ckly_z, g2_ckly_z, g3_ckly_z, g4_ckly_z,
                          g5_ckly_z]
 
        glist_ckl_z = [g1_ckl_z, g2_ckl_z, g3_ckl_z, g4_ckl_z]
                              
        g_az = [glist_cikly_az, glist_cikl_az, glist_ckly_az, glist_ckl_az]                  

        g_z = [glist_cikly_z, glist_cikl_z, glist_ckly_z, glist_ckl_z]                  

        gvec = sympy.Matrix( glist_cikly_az )       
        
        return g_az, g_z
        
        

    def gvec_ss_toymodel_full_info_sympy(self, g_az, g_z):
        
        
        g_az_ss = [self.make_expressions_at_ss_fi(g) for g in g_az]
        
        g_z_ss = [self.make_expressions_at_ss_fi(g) for g in g_z]
        
        return g_az_ss, g_z_ss
        
        
        
        
    def gvec_toymodel_fullinfo_numpy_from_sympy(self):
        
        
        gvec_in_numpy = 1
        return gvec_in_numpy

        
        
    def gvec_toymodel_fi(self, symbolic_elem = 'just_x_w'):
        r"""
        Return a list with the LHS of 
        $g(x_tp1,x_t,x_tm1,w_tp1,w_t)=0$, describing the equilibrium
        conditions of the model and using user-supplied parameter values.
        Only variables in $x$ and $w$ remain as symbols. 
        The main goal is to provide a way to evaluate approximation errors and
        a sympy expression to differentiate w.r.t $x$ and $w$
        """        
        

        if symbolic_elem=='all_but_volatilities':
            alpha, beta = self.alpha_sym, self.beta_sym
            delta, nu = self.delta_sym, self.nu_sym
            A, D = self.A_sym, self.D_sym
            C, G =  self.C, self.G
       
            
        elif symbolic_elem=='just_x_w':
            alpha, beta, delta, nu = self.alpha, self.beta, self.delta, self.nu
            A, C, D, G = self.A, self.C, self.D, self.G
            
 
        elif symbolic_elem=='all':
            alpha, beta = self.alpha_sym, self.beta_sym
            delta, nu = self.delta_sym, self.nu_sym
            A, C, D, G = self.A_sym, self.C_sym, self.D_sym, self.G_sym
       
        
        zt_sym, ztm1_sym, zetaat_sym = self.zt_sym, self.ztm1_sym, self.zetaat_sym
        wt_sym, wtp1_sym = self.wt_sym, self.wtp1_sym
        
        
        Ct, Lt, Kstate =  self.Ct_sym, self.Lt_sym, self.Kstate_sym 
        Kpolicy, Ctp1, Ltp1 =  self.Kpolicy_sym, self.Ctp1_sym, self.Ltp1_sym 
        at = (D*ztm1_sym + G*wt_sym)[0]
        atp1 = (D *zt_sym + G*wtp1_sym)[0]
        
        if self.xvarstring=='CKL':
            # at, atp1, Yt are not primitive objects, but formulas are much shorter
            # if using these intermediate expressions:
            
        
            Yt = sympy.exp((1-alpha)*at)  * Kstate**(alpha) * Lt**(1-alpha) 
            Ytp1 = sympy.exp((1-alpha)*atp1)  * Kpolicy**(alpha) * Ltp1**(1-alpha) 
        
            # Toy model FOCs, under full information
            g1 = beta*sympy.exp(-at) * (Ct/Ctp1) * \
                (alpha * Ytp1/Kpolicy    +(1-delta)) - 1
        
            g2 = Yt - sympy.exp(at) * Kpolicy + (1-delta)*Kstate - Ct 
        
            g3 = nu*(1-alpha)*Yt/(Ct*(1-nu)) - Lt/(1-Lt)
        
            # state space of productivity growth are also equilibrium conditions
            g4 = zetaat_sym - (A*ztm1_sym + C*wt_sym)[1]
            
            glist = [g1, g2, g3, g4]
        
        elif self.xvarstring=='CKLIY':
            # Toy model FOCs, under full information
        
            Yt = self.Yt_sym
            Ytp1 = self.Ytp1_sym
            It = self.It_sym         
            
            g1 = beta*sympy.exp(-at) * (Ct/Ctp1) * \
                (alpha * Ytp1/Kpolicy    +(1-delta)) - 1
        
            g2 = Yt - It - Ct 
        
            g3 = (1-Lt)* ((nu*(1-alpha))/(1-nu)) *Yt/Ct - Lt
            
            g4 = Yt - sympy.exp((1-alpha)*at)  * Kstate**(alpha) * Lt**(1-alpha)
            
            g5 = It - sympy.exp(at) * Kpolicy + (1-delta)*Kstate
        
            # state space of productivity growth are also equilibrium conditions
            g6 = zetaat_sym - (A*ztm1_sym + C*wt_sym)[1]
            
            glist = [g1, g2, g3, g4, g5, g6]
                
        
        if symbolic_elem=='just_x_w':
            glist = [g.subs({self.mua_sym: self.mua, 
                              self.rhoa_sym: self.rhoa}) for g in glist]
        
#        gvec = (g1, g2, g3, g4)       
        gvec = sympy.Matrix( glist )       
        
        
        return gvec
        
        
    def make_expressions_at_ss_fi(self, sympy_object_to_convert):
        r"""
        Return a list with expressions converted to steady state versions 
        """        
        

        #dictionary_with_space_state_values_in_ss = {muass: mua_valueSSFI}
        if isinstance(sympy_object_to_convert, sympy.Matrix):
            sympy_object_in_ss = sympy_object_to_convert.subs(
                self.dictionary_with_steady_state_versions)
            
        elif isinstance(sympy_object_to_convert, (list,tuple)):
            sympy_object_in_ss = [
               elem.subs(self.dictionary_with_steady_state_versions) for 
               elem in sympy_object_to_convert]      
            
        return sympy_object_in_ss 
        
        
    def get_steady_state_x_values(self):
        
        
        if self.ss_solutions_dict == {}:
            self.update_ss_solutions_fi_toy()
            
        
        vals_x_in_ss_array = np.empty((6,1))
        
        
#        print self.ss_solutions_dict
        
#        print self.xvar_atss_sym
        
        vals_x_in_ss_list = [x.subs(self.ss_solutions_dict) for x in
                        self.xvar_atss_sym]
                
        vals_x_in_ss_array[:,0] = np.array(vals_x_in_ss_list)
        
        return vals_x_in_ss_array
        
    def get_steady_state_x_dict(self):
        
        if self.ss_solutions_dict == {}:
            self.update_ss_solutions_fi_toy()
            
        return self.ss_solutions_dict       
        
        
    def update_ss_solutions_fi_toy(self):
          
#        print ' \n Hey! calculating steady state (again?) \n'
        
        oldxvs = self.xvarstring
        self.xvarstring = 'CKL'
        gvec = self.gvec_toymodel_fi()
        self.xvarstring = oldxvs
        
                                
        gvec_ss = self.make_expressions_at_ss_fi(gvec)
        
        # the last equation is identically zero, so we drop it
        gvec_ss_for_solver =  gvec_ss[0:-1]
  
       # print gvec_ss
#        print gvec_ss_for_solver  
        
        if self.modeltype == 'toy':
        # Now, compare it to the closed form solution
            B1, B3, B4, B5 = sympy.symbols('B1, B3, B4, B5')

            B_dictionary = {B1: self.beta_sym * sympy.exp(-self.mua_sym),
                            B3: (1-self.alpha_sym)*self.nu_sym/(1-self.nu_sym),
                            B4: self.beta_sym * sympy.exp(-self.mua_sym),
                            B5: sympy.exp((1-self.alpha_sym)*self.mua_sym)}
            
            TexSolYtoKatSS = (sympy.exp(self.mua_sym) - self.beta_sym* \
                             (1-self.delta_sym))/(self.alpha_sym * self.beta_sym)
                             
            TexSolL = B3*TexSolYtoKatSS/ \
                     (1- self.delta_sym - sympy.exp(self.mua_sym) +  (1+B3)*TexSolYtoKatSS )

            TexSolK = sympy.exp(self.mua_sym)*self.Lss_sym* \
                      TexSolYtoKatSS**(1/(self.alpha_sym-1))

            TexSolY = self.Kss_sym * TexSolYtoKatSS

            TexSolC = self.Kss_sym*(TexSolYtoKatSS - sympy.exp(self.mua_sym) \
                      + 1 - self.delta_sym)
            
#            YtoKatSS_evauated = \
#                TexSolYtoKatSS.subs(self.dictionary_sym2numval_params)
#                
            Lss_hand_sol_evauated = TexSolL.subs(
                B_dictionary).subs(self.dictionary_sym2numval_params)

            Kss_hand_sol_evauated = TexSolK.subs(
                {self.Lss_sym: Lss_hand_sol_evauated}).subs(
                self.dictionary_sym2numval_params)

#            Yss_hand_sol_evauated = TexSolY.subs(
#                {self.Kss_sym: Kss_hand_sol_evauated}).subs(
#                self.dictionary_sym2numval_params)

            Css_hand_sol_evauated = TexSolC.subs(
                {self.Kss_sym: Kss_hand_sol_evauated}).subs(
                self.dictionary_sym2numval_params)
                
            YsstoKss_hand_sol_evaluated = TexSolYtoKatSS.subs(
                {self.Kss_sym: Kss_hand_sol_evauated}).subs(
                self.dictionary_sym2numval_params)    
            
            Yss_hand_sol_evaluated = YsstoKss_hand_sol_evaluated* \
                                        Kss_hand_sol_evauated
            
            dictionary_solutions_SS_hand_formulas = {
                self.Css_sym:Css_hand_sol_evauated,
                self.Kss_sym:Kss_hand_sol_evauated,
                self.Lss_sym:Lss_hand_sol_evauated,
                self.Yss_sym:Yss_hand_sol_evaluated}
                
                
#            print(dictionary_solutions_SS_hand_formulas)
           
#            print 'equations evaluated at hand sols:'
            eqsathandsolve = [
                gvec_ss_for_solver[i].subs(dictionary_solutions_SS_hand_formulas) \
                for i in range(len(gvec_ss_for_solver))]
                    
#            print eqsathandsolve
        
#            return gvec_ss_for_solver         


        
#        sol=sympy.nsolve(gvec_ss_for_solver,
#                         (self.Css_sym, self.Kss_sym, self.Lss_sym), (0.6, 2, 0.5))

        if self.xvarstring=='CKL':
            sol=sympy.nsolve(gvec_ss_for_solver[0:3], 
                         self.xvar_atss_sym, (0.6, 2, 0.5))
            dictionary_solutions_SS_nsolve  =dict(zip(
               [self.Css_sym,self.Kss_sym,self.Lss_sym], sol))
               
        elif self.xvarstring=='CKLIY':
#            print 'gvec_ss_for_solver[0:3]',gvec_ss_for_solver[0:3]
            sol_ckl = sympy.nsolve(gvec_ss_for_solver[0:3], 
                         (self.Css_sym, self.Kss_sym, self.Lss_sym), (0.6, 2, 0.5))
            ckl_dict = dict(zip(
               [self.Css_sym,self.Kss_sym,self.Lss_sym], sol_ckl))
            eqi =   sympy.exp(self.mua) * self.Kss_sym -  (1-self.delta)*self.Kss_sym    
            sol_i = eqi.subs(ckl_dict)
            
            sol_y_ci = self.Css_sym.subs(ckl_dict) +  sol_i
            
            iy_dict = {self.Iss_sym:sol_i, self.Yss_sym:sol_y_ci}
            
            dictionary_solutions_SS_nsolve = ckl_dict
            dictionary_solutions_SS_nsolve.update(iy_dict)
            
##            eqy = sympy.exp((1-self.alpha)*self.mua)  * self.Kss_sym**(self.alpha) * self.Lss_sym**(1-self.alpha)
#            
##            sol_y_kl = eqy.subs(ckl_dict)
#            
#            print 'sol_i', sol_i
#            print 'sol_y_ci', sol_y_ci
##            print 'sol_y_kl', sol_y_kl
#                         
                         
                         
        
        
               
#        print(dictionary_solutions_SS_nsolve)
        
#        print 'equations evaluated at nsolve sols:'
        eqsatnsolve = [
            gvec_ss_for_solver[i].subs(dictionary_solutions_SS_nsolve) \
            for i in range(len(gvec_ss_for_solver))]
                    
#        print eqsatnsolve
        
        
        self.ss_solutions_dict = dictionary_solutions_SS_nsolve     
        self.ss_solutions_dict.update({self.zetaass_sym:0})
        self.ss_solutions_dict.update({self.muass_sym:self.mua})
 
            
    def gvec_autodiff_at_ss_toymodel_fi(self, symbolic_elem='just_x_w', 
                                        subs_x_to_ss_num=True):
        sym_choice = symbolic_elem
        
        gvec = self.gvec_toymodel_fi(symbolic_elem = sym_choice)
        
        
        
    def gvec_diffs_ev_at_ss_toymodel_fi(self, symbolic_elem='just_x_w', 
                                        subs_x_to_ss_num=True):
        
        sym_choice = symbolic_elem
        
        gvec = self.gvec_toymodel_fi(symbolic_elem = sym_choice)
        
        deriv_vars_x_t = self.xvar_t_sym

        deriv_vars_x_tm1 = self.xvar_tm1_sym

        deriv_vars_x_tp1 = self.xvar_tp1_sym
        
        deriv_vars_w_t = self.wvar_t_sym
        
        deriv_vars_w_tp1 = self.wvar_tp1_sym
        
        
#        deriv_vars_x_t = [self.Con_sym, self.Kpolicy_sym, self.Lt_sym,
#                            self.zetaat_sym]
#
#        deriv_vars_x_tm1 = [self.Ctm1_sym, self.Kstate_sym, self.Ltm1_sym,
#                            self.zetaatm1_sym]
#
#        deriv_vars_x_tp1 = [self.Ctp1_sym, self.Ktp2_sym, self.Ltp1_sym,
#                            self.zetaatp1_sym]
#                            

        deriv_vars_x_t = self.xvar_t_sym

        deriv_vars_x_tm1 = self.xvar_tm1_sym

        deriv_vars_x_tp1 = self.xvar_tp1_sym
        
        deriv_vars_w_t = self.wvar_t_sym
        
        deriv_vars_w_tp1 = self.wvar_tp1_sym
        

#        deriv_vars_w_t = [self.wat_sym, self.wzetaat_sym]
#
#        deriv_vars_w_tp1 = [self.watp1_sym, self.wzetaatp1_sym]
#        
##        neqs = len(gvec)
##        nx = len(deriv_vars_x_t)
##        
##        gvec_dx = [sympy.diff(gvec[i], deriv_vars_x_t[j]) for i in range(neqs)\
##                   for j in range(nx)]
        
        gvec_d_xt = gvec.jacobian(deriv_vars_x_t)
        gvec_d_xt_at_ss = self.make_expressions_at_ss_fi( gvec_d_xt)
        
        gvec_d_xtm1 = gvec.jacobian(deriv_vars_x_tm1)
        gvec_d_xtm1_at_ss = self.make_expressions_at_ss_fi( gvec_d_xtm1)
        
        gvec_d_xtp1 = gvec.jacobian(deriv_vars_x_tp1)
        gvec_d_xtp1_at_ss = self.make_expressions_at_ss_fi( gvec_d_xtp1)
        
        gvec_d_wt = gvec.jacobian(deriv_vars_w_t)
        gvec_d_wt_at_ss = self.make_expressions_at_ss_fi( gvec_d_wt)
        
        gvec_d_wtp1 = gvec.jacobian(deriv_vars_w_tp1)
        gvec_d_wtp1_at_ss = self.make_expressions_at_ss_fi( gvec_d_wtp1)
        

#        print "gvec", gvec
#        print "deriv_vars_w_tp1", deriv_vars_w_tp1 
#        print "gvec_d_wtp1", gvec_d_wtp1
        
        
        
        if subs_x_to_ss_num==True:
            
            if self.ss_solutions_dict == {}:
                self.update_ss_solutions_fi_toy()
            gvec_d_xtm1_at_ss = gvec_d_xtm1_at_ss.subs(self.ss_solutions_dict)
            gvec_d_xt_at_ss = gvec_d_xt_at_ss.subs(self.ss_solutions_dict)
            gvec_d_xtp1_at_ss = gvec_d_xtp1_at_ss.subs(self.ss_solutions_dict)
            gvec_d_wt_at_ss = gvec_d_wt_at_ss.subs(self.ss_solutions_dict)
            gvec_d_wtp1_at_ss = gvec_d_wtp1_at_ss.subs(self.ss_solutions_dict)
            
            
            gvec_d_xtm1_at_ss = self.matrix2numpyfloat(gvec_d_xtm1_at_ss)
            gvec_d_xt_at_ss = self.matrix2numpyfloat(gvec_d_xt_at_ss)
            gvec_d_xtp1_at_ss = self.matrix2numpyfloat(gvec_d_xtp1_at_ss) 
            gvec_d_wt_at_ss = self.matrix2numpyfloat(gvec_d_wt_at_ss)
            gvec_d_wtp1_at_ss = self.matrix2numpyfloat(gvec_d_wtp1_at_ss)        

              
#        return  np.asarray(gvec_d_xtm1_at_ss), np.asarray(gvec_d_xt_at_ss), \
#                np.asarray(gvec_d_xtp1_at_ss), np.asarray(gvec_d_wt_at_ss), \
#                np.asarray(gvec_d_wtp1_at_ss)
              
        return  gvec_d_xtm1_at_ss, gvec_d_xt_at_ss, \
                gvec_d_xtp1_at_ss, gvec_d_wt_at_ss, \
                gvec_d_wtp1_at_ss              
                
                
    
    
#    def get_first_order_approx_coeff(self, gxtm1_ss, gx_ss, gxtp1_ss):
        
    def get_first_order_approx_coeff(self,isPartialInfo=False, isloglog=False):
            
#        gxtm1_ss, gxt_ss, gxtp1_ss, gwt_ss, gwtp1_ss =  \
#            self.gvec_diffs_ev_at_ss_toymodel_fi()     
            
#        
        if isPartialInfo:
            gxtm1_ss,gx_ss,gxtp1_ss,gwt_ss,gwtp1_ss = self.gvec_diffs_ev_at_ss_toymodel_pi()
        else:
            gxtm1_ss,gx_ss,gxtp1_ss,gwt_ss,gwtp1_ss = self.gvec_diffs_ev_at_ss_toymodel_fi()
        
                  
        nx = gxtp1_ss.shape[1] # it returns the number of columns of A
             
        #quad_coeffmat_in_eq_for_P is Uhlig's \Psi matrix, sensible
#        quad_coeffmat_in_eq_for_P = self.matrix2numpyfloat(gxtp1_ss)
        quad_coeffmat_in_eq_for_P = gxtp1_ss
        
        #lin_coeffmat_in_eq_for_P is Uhlig's (-\Gamma) matrix, sensible
#        lin_coeffmat_in_eq_for_P = self.matrix2numpyfloat(gx_ss)
        lin_coeffmat_in_eq_for_P = gx_ss
        
        #cons_coeffmat_in_eq_for_P is Uhlig's (-\Theta)\Psi matrix, sensible
        cons_coeffmat_in_eq_for_P =  gxtm1_ss
#        cons_coeffmat_in_eq_for_P =  self.matrix2numpyfloat(gxtm1_ss)
#
       
        nnzero = np.zeros((nx,nx))

        # this is Uhlig's Chi matrix:
        lin_con_mat = np.bmat(
            [[-lin_coeffmat_in_eq_for_P, -cons_coeffmat_in_eq_for_P] ,
                                  [np.identity(nx), nnzero] ])  
        
        # this is Uhlig's Delta matrix:                 
        quad_mat = np.bmat(
            [[quad_coeffmat_in_eq_for_P, nnzero] ,[nnzero, np.identity(nx)]])
            
            
        stable_phi_x = self.solve_quad_matrix_stable_sol_QZ(lin_con_mat,
                                                        quad_mat, nx)    
                                        
        phi_w_inner = np.dot(gxtp1_ss, stable_phi_x) + gx_ss
        phi_w = - np.dot(np.linalg.inv(phi_w_inner), gwt_ss)
        
        if isloglog == True:
            linlinco = np.hstack((stable_phi_x, phi_w))
            vss = self.get_steady_state_x_values()
            sel_vec = np.array([True, True, True, False, True, False,
                                  False, False]) 
            v_allss_xw = np.vstack((vss,np.zeros((3,1))))
            loglog_coeffs = self.from_linlin_to_loglog(linlinco, v_allss_xw,
                                                       sel_vec)
            stable_phi_x = loglog_coeffs[:,0:6]
            if isPartialInfo==True:
                phi_w = loglog_coeffs[:,6:7]
            else:
                phi_w = loglog_coeffs[:,6:8]
                            
        
#        return stable_phi_x, phi_w
        return stable_phi_x, phi_w
    
        
        
    def solve_quad_matrix_stable_sol_QZ(self, lin_con_mat, quad_mat, nx):
        
    
        # lin_con_mat is Uhlig's toolkit \Xi matrix
        # quad_mat is Uhlig's toolkit \Delta matrix
        
#        print '\n'
#        print 'type(lin_con_mat):' ,type(lin_con_mat)         
#        print '\n'
#        print 'type(quad_mat):' ,type(quad_mat)         
        
        
        
#        gqzresults = rgqz(np.asarray(lin_con_mat), np.asarray(quad_mat),
#                            sort="S")

        gqzresults = rgqz(np.asarray(lin_con_mat), np.asarray(quad_mat),
                            sort="S")

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
        print [self.wat_sym, self.wzetaat_sym]
        
    
    def impulse_response_toy_fi(self, phi_x, phi_w, shock = 1, x0 = [],
                                finalT = 30, shock_type='wa', shock_mag=1,
                                dologs=False):
        
            

#        print 'phi_x.shape',phi_x.shape
#        print 'phi_x.shape[0]',phi_x.shape[0]
        
        steady_s_list_sym = [self.Css_sym, self.Kss_sym, self.Lss_sym,
                         self.Iss_sym, self.Yss_sym]     
                         
        steady_s_list_num = [x.subs(self.ss_solutions_dict) for x 
                            in steady_s_list_sym]
                            
        
        steady_s_vec = np.zeros((6,1))
        steady_s_vec[0:-1,:] = np.array([steady_s_list_num]).T

        
#        print 'steady_s_list_sym', steady_s_list_sym
#        print 'steady_s_list_num', steady_s_list_num
#        print 'steady_s_vec', steady_s_vec
#        
        
        if x0 == []:
            x0 = np.zeros( (phi_x.shape[0],1))
        
        x_time_series_mat = np.empty( (x0.shape[0], finalT))
        
#        print '\n'
#        print 'x_time_series_mat.shape', x_time_series_mat.shape
        
        
        x_non_sta_time_series_mat = np.empty( (x0.shape[0], finalT))
        
        azA_mat = np.empty( (3, finalT))
        
        
        mua = self.mua
        zetaa_minusone = 0
        current_x = x0
                
        A_minusone = 1.0
        
        impulse_vector = np.zeros((2,1))
        
        if shock_type=='wa':
            impulse_vector[0,0] = shock_mag
        elif shock_type=='wzetaa':
            impulse_vector[1,0] = shock_mag
        
        
        if dologs:
            linlinco = np.hstack((phi_x, phi_w))
            vss = self.get_steady_state_x_values()
            sel_vec = np.array([True, True, True, False, True, False,
                                  False, False]) 
            v_allss_xw = np.vstack((vss,np.zeros((3,1))))
            loglog_coeffs = self.from_linlin_to_loglog(linlinco, v_allss_xw,
                                                       sel_vec)
            phi_x = loglog_coeffs[:,0:6]
            phi_w = loglog_coeffs[:,6:8]
#            steady_s_vec = np.vstack((vss,np.zeros((1,1)))) #only x variables
            steady_s_vec[sel_vec] = np.log(steady_s_vec[sel_vec])
                        
        
            
        first_response_x = np.dot(phi_x, x0) + np.dot(phi_w, impulse_vector)
        
#        print 'steady_s_vec line 942', steady_s_vec       
#        print 'first_response_x',  first_response_x
        
        first_new_values_x = steady_s_vec +  first_response_x
        
        x_time_series_mat[:,0] = first_new_values_x[:,0]   
#        x_non_sta_time_series_mat[:,0] = first_response_x[:,0]/A_minusone 
        x_non_sta_time_series_mat[:,0] = first_new_values_x[:,0]*A_minusone 
        
        a_zero = np.dot(self.D, np.array( (mua, zetaa_minusone))) + \
                    np.dot(self.G,impulse_vector)        
        
        zetaa_zero = self.rhoa*zetaa_minusone + \
                        np.dot(self.C,impulse_vector)[1,:]
      
        A_zero = A_minusone*np.exp(a_zero)        
        
        azA_mat[0,0] = a_zero # a_0 has the direct impact of the impulse
        azA_mat[1,0] = zetaa_zero # z_{0} has the direct impact of the impulse  
        azA_mat[2,0] = A_minusone # A_{0} has the direct impact of the impulse
        
        current_zetaa = zetaa_zero
        current_A = A_zero
        
        current_x = first_new_values_x # C_0, K_1, L_0 etc.
        
#        print '\n'
#        print 'steady_s_vec', steady_s_vec
                
        for t in range(finalT-1):  
            current_x = steady_s_vec + np.dot(phi_x, current_x-steady_s_vec)
            x_time_series_mat[:,t+1] = current_x[:,0] # C_1, K_2, L_1 ...
            # \tilde{C}_1, \tilde{K}_2, L_1 ...
#            x_non_sta_time_series_mat[:,t+1] = current_x[:,0]/current_A
            
            if dologs:
                x_non_sta_time_series_mat[:,t+1] = current_x[:,0] +  np.log(current_A) 
            else:
                x_non_sta_time_series_mat[:,t+1] = current_x[:,0]*current_A
            
            current_a = np.dot(self.D, np.array( (mua, current_zetaa))) # those are a_1 and z_0
            
            
            current_zetaa = self.rhoa*current_zetaa # z_1 and z_0
            
            azA_mat[0,t+1] = current_a  # a_1, a_2 ...
            azA_mat[1,t+1] = current_zetaa  # z_1, z_2 
            azA_mat[2,t+1] = current_A  # A_0, A_1
            
            current_A = current_A*np.exp(current_a) # A_1, A_0, a_1
        
        x_non_sta_time_series_mat[2,:] = x_time_series_mat[2,:]
        
        x_non_sta_time_series_mat[-1,:] = x_time_series_mat[-1,:]
        
        BGP_det = np.zeros(x_non_sta_time_series_mat.shape)
        for i in range(BGP_det.shape[0]):
            BGP_det[i,:] = azA_mat[2,:]*steady_s_vec[i]
            
        return azA_mat, x_time_series_mat, x_non_sta_time_series_mat, BGP_det
    
        
        
    def matrix2numpyfloat(self, m):
        a = np.empty(m.shape, dtype='float')
        for i in range(m.rows):
            for j in range(m.cols):
                a[i,j] = m[i,j]
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
                coeffs_of_log_log[i,j] = ratio_ss*coeffs_of_linlin[i,j]
                
        return coeffs_of_log_log
            
    
    
         
            
class PartialInfoModel(FullInfoModel):
    x = 0
    def __init__(self, statespace,  alpha = 0.33, delta = 0.1, gamma=1,
                 beta = 0.95, rhoa=0.95, nu=0.5, muatrue = 0.03, 
                 modeltype = 'toy', statespace_symdict = {},
                 ss_x_dict={}, xvarstring='CKL', prior_mean=[], prior_variance=[]):
                     
        FullInfoModel.__init__(self, statespace, alpha, delta, gamma, beta, rhoa,
                         nu, muatrue, modeltype, statespace_symdict, ss_x_dict,
                         xvarstring)
                         
        self.z_initial = prior_mean
        self.Sigma_initial = prior_variance
        
        self.current_z = self.z_initial
        self.current_Sigma = self.Sigma_initial
        
        self.add_model_symbols_pi()
                
        self.zetaat_sym_hat = self.modelsymbols_dict['zetaat_hat']
        self.zetaatm1_sym_hat = self.modelsymbols_dict['zetaatm1_hat']
        self.zetaatp1_sym_hat = self.modelsymbols_dict['zetaatp1_hat']
        
        self.zetaa_ss_sym_hat = self.modelsymbols_dict['zetaa_ss_hat']
        
        
        self.zt_sym_hat = self.modelsymbols_dict['zt_hat']
        self.ztp1_sym_hat = self.modelsymbols_dict['ztp1_hat']
        self.ztm1_sym_hat = self.modelsymbols_dict['ztm1_hat']
        
        self.inno_at_sym = self.modelsymbols_dict['inno_at'] 
        self.inno_atp1_sym = self.modelsymbols_dict['inno_atp1'] 
        
        self.innot_sym = self.modelsymbols_dict['innot'] 
        self.innotp1_sym = self.modelsymbols_dict['innotp1'] 
        
        self.KalG_mua_sym  = self.modelsymbols_dict['KalG_mua']
        self.KalG_zetaa_sym  = self.modelsymbols_dict['KalG_zetaa']
        self.KalG_sym = self.modelsymbols_dict['KalG']
        
        self.KalG_ss_mua_sym  = self.modelsymbols_dict['KalG_ss_mua']
        self.KalG_ss_zetaa_sym  = self.modelsymbols_dict['KalG_ss_zetaa']
        self.KalG_ss_sym = self.modelsymbols_dict['KalG_ss']
        
        self.Sigma_ss, self.KalG_ss = self.get_stationary_Sigma_Kgain()
        
        if self.xvarstring=='CKL':
            self.xvar_tm1_sym = self.xvar_tm1_sym[0:-1]
            self.xvar_tm1_sym.append(self.zetaatm1_sym_hat)
            
            self.xvar_t_sym = self.xvar_t_sym[0:-1]
            self.xvar_t_sym.append(self.zetaat_sym_hat)
            
            self.xvar_tp1_sym = self.xvar_tp1_sym[0:-1]
            self.xvar_tp1_sym.append(self.zetaatp1_sym_hat)
            
            self.xvar_atss_sym = [self.Css_sym, self.Kss_sym, self.Lss_sym,
                                  self.zetaass_sym]
            
        elif self.xvarstring=='CKLIY':
            self.xvar_tm1_sym = self.xvar_tm1_sym[0:-1]
            self.xvar_tm1_sym.append(self.zetaatm1_sym_hat)
            
            self.xvar_t_sym = self.xvar_t_sym[0:-1]
            self.xvar_t_sym.append(self.zetaat_sym_hat)
            
            self.xvar_tp1_sym = self.xvar_tp1_sym[0:-1]
            self.xvar_tp1_sym.append(self.zetaatp1_sym_hat)
            
            self.xvar_atss_sym = [self.Css_sym, self.Kss_sym, self.Lss_sym,
                                  self.Iss_sym, self.Yss_sym, self.zetaass_sym]
                                  

        self.wvar_t_sym = [self.inno_at_sym]

        self.wvar_tp1_sym = [self.inno_atp1_sym] 
        
                              
        additional_ss_dict = {self.zetaatm1_sym_hat: 0, 
                              self.zetaat_sym_hat: 0,
                              self.zetaatp1_sym_hat: 0,
                              self.inno_at_sym:0, self.inno_atp1_sym:0}
  

#        additional_ss_dict = {self.zetaatm1_sym_hat: self.zetaa_ss_sym_hat, 
#                              self.zetaat_sym_hat: self.zetaa_ss_sym_hat,
#                              self.zetaatp1_sym_hat: self.zetaa_ss_sym_hat,
#                              self.inno_at_sym:0, self.inno_atp1_sym:0}
#  
        
                    
        self.dictionary_with_steady_state_versions.update(additional_ss_dict)


            
            
            
                
        
    def add_model_symbols_pi(self):
        
        zetaat_sym_hat = sympy.Symbol('\check{\zeta}_at', real=True)
        zetaatm1_sym_hat = sympy.Symbol('\check{\zeta}_atm1', real=True)
        zetaatp1_sym_hat = sympy.Symbol('\check{\zeta}_atp1', real=True)
        
        zetaa_ss_sym_hat = sympy.Symbol('\check{\zeta}_{a,ss}', real=True)
        
        zt_sym_hat = sympy.Matrix([[self.mua_sym],[zetaat_sym_hat]])
        ztp1_sym_hat = sympy.Matrix([[self.mua_sym],[zetaatp1_sym_hat]])
        ztm1_sym_hat = sympy.Matrix([[self.mua_sym],[zetaatm1_sym_hat]])
        
        inno_at_sym = sympy.Symbol('i_at', real=True) 
        inno_atp1_sym = sympy.Symbol('i_atp1', real=True) 
        
        innot_sym = sympy.Matrix([[inno_at_sym]])
        innotp1_sym = sympy.Matrix([[inno_atp1_sym]])
        
        KalG_mua_sym  = sympy.Symbol('KG_{\mu_a}', real=True)
        KalG_zetaa_sym  = sympy.Symbol('KG_{\zeta_a}', real=True)
        KalG_sym = sympy.Matrix([[KalG_mua_sym],[KalG_zetaa_sym]])
        
        KalG_ss_mua_sym  = sympy.Symbol('KG_{\mu_a,ss}', real=True)
        KalG_ss_zetaa_sym  = sympy.Symbol('KG_{\zeta_a,ss}', real=True)
        KalG_ss_sym = sympy.Matrix([[KalG_ss_mua_sym],[KalG_ss_zetaa_sym]])
        
        
        
        additional_symbols_dict = {'KalG':KalG_sym, 'KalG_mua':KalG_mua_sym, 
            'KalG_zetaa':KalG_zetaa_sym, 'KalG_ss':KalG_ss_sym,
            'KalG_ss_mua':KalG_ss_mua_sym, 'KalG_ss_zetaa':KalG_ss_zetaa_sym,
            'zt_hat':zt_sym_hat, 'ztm1_hat':ztm1_sym_hat,
            'ztp1_hat':ztp1_sym_hat, 'zetaat_hat':zetaat_sym_hat,
            'zetaatm1_hat':zetaatm1_sym_hat, 'zetaatp1_hat':zetaatp1_sym_hat,
            'zetaa_ss_hat':zetaa_ss_sym_hat,
            'inno_at':inno_at_sym, 'inno_atp1':inno_atp1_sym,
            'innot':innot_sym, 'innotp1':innotp1_sym}
            
        
        self.modelsymbols_dict.update(additional_symbols_dict)
        
        
        
        
        
        
    def update_z_Sigma(self, s_obs):
        innovation = s_obs - np.dot(self.D, self.current_z)

        kg_ASDt = np.dot(self.A, np.dot(self.current_Sigma, self.D.T))
        kg_DSDt = np.dot(np.dot(self.D, self.current_Sigma), self.D.T)
        GGt = np.dot(self.G, self.G.T)
        DSDGGi = scipy.linalg.inv(kg_DSDt+GGt)
        kalman_gain = kg_ASDt - DSDGGi
        
        ASAt = np.dot(np.dot(self.A, self.current_Sigma), self.A.T)   
        KD = np.dot(kalman_gain,self.D)
        SAt = np.dot(self.current_Sigma, self.A.T)
        KDSAt = np.dot(KD,SAt)
        next_Sigma = ASAt - KDSAt + np.dot(self.C, self.C.T)
        
        next_z = np.dot(self.A,self.current_z) + np.dot(kalman_gain,innovation)
                
        self.current_Sigma = next_Sigma
        self.current_z = next_z
        
        
    def get_stationary_Sigma_Kgain(self):
       
         # === simplify notation === #
        A, D = self.A, self.D
        R = np.dot(self.G, self.G.T)
        Q = np.dot(self.C, self.C.T)
       
        # === solve Riccati equation, obtain Kalman gain === #
        Sigma_infinity = riccati.dare(A.T, D.T, R, Q)
        temp1 = np.dot(np.dot(A, Sigma_infinity), D.T)
        temp2 = scipy.linalg.inv(np.dot(D, np.dot(Sigma_infinity, D.T)) + R)
        K_infinity = np.dot(temp1, temp2)
        return Sigma_infinity, K_infinity
        
        
    
    
    def update_ss_solutions_pi_toy(self, current_z):
        
        store_mua = self.mua 
        if self.ss_solutions_dict == {}:
                self.update_ss_solutions_fi_toy()
                
#        print '\n  self.ss_solutions_dict \n', self.ss_solutions_dict
#
#        print '\n current_z \n', current_z
        
        self.mua = current_z[0]
        
        self.update_ss_solutions_fi_toy()
        self.mua = store_mua
    
    
        
    def gvec_toymodel_pi(self, symbolic_elem = 'just_x_w'):
        r"""
        Return a list with the LHS of 
        $g(x_tp1,x_t,x_tm1,w_tp1,w_t)=0$, describing the equilibrium
        conditions of the model and using user-supplied parameter values.
        Only variables in $x$ and $w$ remain as symbols. 
        The main goal is to provide a way to evaluate approximation errors and
        a sympy expression to differentiate w.r.t $x$ and $w$
        """        
        

        if symbolic_elem=='all_but_volatilities':
            alpha, beta = self.alpha_sym, self.beta_sym
            delta, nu = self.delta_sym, self.nu_sym
            A, D = self.A_sym, self.D_sym
            C, G =  self.C, self.G
            KalG_ss = self.KalG_ss
       
            
        elif symbolic_elem=='just_x_w':
            alpha, beta, delta, nu = self.alpha, self.beta, self.delta, self.nu
            A, C, D, G = self.A, self.C, self.D, self.G
            KalG_ss = self.KalG_ss
            
 
        elif symbolic_elem=='all':
            alpha, beta = self.alpha_sym, self.beta_sym
            delta, nu = self.delta_sym, self.nu_sym
            A, C, D, G = self.A_sym, self.C_sym, self.D_sym, self.G_sym
            KalG_ss = self.KalG_ss_sym
        
       
        
        zt_sym_hat, ztm1_sym_hat = self.zt_sym_hat, self.ztm1_sym_hat
        zetaat_sym_hat = self.zetaat_sym_hat
        
        innot_sym, innotp1_sym = self.innot_sym, self.innotp1_sym
        
        
        Ct, Lt, Kstate =  self.Ct_sym, self.Lt_sym, self.Kstate_sym 
        Kpolicy, Ctp1, Ltp1 =  self.Kpolicy_sym, self.Ctp1_sym, self.Ltp1_sym 
       
       
       
        at = (D*ztm1_sym_hat + innot_sym)[0]
        atp1 = (D *zt_sym_hat + innotp1_sym)[0]
        
        if self.xvarstring=='CKL':
            # at, atp1, Yt are not primitive objects, but formulas are much shorter
            # if using these intermediate expressions:
            
        
            Yt = sympy.exp((1-alpha)*at)  * Kstate**(alpha) * Lt**(1-alpha) 
            Ytp1 = sympy.exp((1-alpha)*atp1)  * Kpolicy**(alpha) * Ltp1**(1-alpha) 
        
            # Toy model FOCs, under full information
            g1 = beta*sympy.exp(-at) * (Ct/Ctp1) * \
                (alpha * Ytp1/Kpolicy    +(1-delta)) - 1
        
            g2 = Yt - sympy.exp(at) * Kpolicy + (1-delta)*Kstate - Ct 
        
            g3 = nu*(1-alpha)*Yt/(Ct*(1-nu)) - Lt/(1-Lt)
        
            # state space of productivity growth are also equilibrium conditions
            g4 = zetaat_sym_hat - (A*ztm1_sym_hat + KalG_ss*innot_sym)[1]
            
            glist = [g1, g2, g3, g4]
        
        elif self.xvarstring=='CKLIY':
            # Toy model FOCs, under full information
        
            Yt = self.Yt_sym
            Ytp1 = self.Ytp1_sym
            It = self.It_sym         
            
            g1 = beta*sympy.exp(-at) * (Ct/Ctp1) * \
                (alpha * Ytp1/Kpolicy    +(1-delta)) - 1
        
            g2 = Yt - It - Ct 
        
            g3 = (nu/(1-nu))*(1-alpha)*Yt/Ct - Lt/(1-Lt)
            
            g4 = Yt - sympy.exp((1-alpha)*at)  * Kstate**(alpha) * Lt**(1-alpha)
            
            g5 = It - sympy.exp(at) * Kpolicy + (1-delta)*Kstate
        
            # state space of productivity growth are also equilibrium conditions
            g6 = zetaat_sym_hat - (A*ztm1_sym_hat + KalG_ss*innot_sym)[1]
            
            glist = [g1, g2, g3, g4, g5, g6]
                
        
        if symbolic_elem=='just_x_w':
            glist = [g.subs({self.mua_sym: self.mua, 
                              self.rhoa_sym: self.rhoa}) for g in glist]
        
#        gvec = (g1, g2, g3, g4)       
        gvec = sympy.Matrix( glist )       
        
        return gvec


    def gvec_diffs_ev_at_ss_toymodel_pi(self,subs_x_to_ss_num=True):
            

        gvec = self.gvec_toymodel_pi()
        
        deriv_vars_x_t = self.xvar_t_sym

        deriv_vars_x_tm1 = self.xvar_tm1_sym

        deriv_vars_x_tp1 = self.xvar_tp1_sym
        
        deriv_vars_w_t = self.wvar_t_sym

        deriv_vars_w_tp1 = self.wvar_tp1_sym
        
        
        gvec_d_xt = gvec.jacobian(deriv_vars_x_t)
        gvec_d_xt_at_ss = self.make_expressions_at_ss_fi( gvec_d_xt)
        
        gvec_d_xtm1 = gvec.jacobian(deriv_vars_x_tm1)
        gvec_d_xtm1_at_ss = self.make_expressions_at_ss_fi( gvec_d_xtm1)
        
        gvec_d_xtp1 = gvec.jacobian(deriv_vars_x_tp1)
        gvec_d_xtp1_at_ss = self.make_expressions_at_ss_fi( gvec_d_xtp1)
        
        gvec_d_wt = gvec.jacobian(deriv_vars_w_t)
        gvec_d_wt_at_ss = self.make_expressions_at_ss_fi( gvec_d_wt)
        
        gvec_d_wtp1 = gvec.jacobian(deriv_vars_w_tp1)
        gvec_d_wtp1_at_ss = self.make_expressions_at_ss_fi( gvec_d_wtp1)
        
        if subs_x_to_ss_num==True:
            if self.ss_solutions_dict == {}:
                self.update_ss_solutions_pi_toy(self.current_z)
            gvec_d_xtm1_at_ss = gvec_d_xtm1_at_ss.subs(self.ss_solutions_dict)
            gvec_d_xt_at_ss = gvec_d_xt_at_ss.subs(self.ss_solutions_dict)
            gvec_d_xtp1_at_ss = gvec_d_xtp1_at_ss.subs(self.ss_solutions_dict)
            gvec_d_wt_at_ss = gvec_d_wt_at_ss.subs(self.ss_solutions_dict)
            gvec_d_wtp1_at_ss = gvec_d_wtp1_at_ss.subs(self.ss_solutions_dict)
            
#            print 'gvec_d_xtm1_at_ss', gvec_d_xtm1_at_ss            
            
            gvec_d_xtm1_at_ss = self.matrix2numpyfloat(gvec_d_xtm1_at_ss)
            gvec_d_xt_at_ss = self.matrix2numpyfloat(gvec_d_xt_at_ss)
            gvec_d_xtp1_at_ss = self.matrix2numpyfloat(gvec_d_xtp1_at_ss) 
            gvec_d_wt_at_ss = self.matrix2numpyfloat(gvec_d_wt_at_ss)
            gvec_d_wtp1_at_ss = self.matrix2numpyfloat(gvec_d_wtp1_at_ss)        
              
        return  gvec_d_xtm1_at_ss, gvec_d_xt_at_ss, \
                gvec_d_xtp1_at_ss, gvec_d_wt_at_ss, \
                gvec_d_wtp1_at_ss              
                

    
    def impulse_response_toy_pi(self, phi_x, phi_w, shock = 1, x0 = [],
                                finalT = 30, shock_type='wa', shock_mag=1,
                                dologs=False):

           
#        print 'phi_x.shape',phi_x.shape
#        print 'phi_x.shape[0]',phi_x.shape[0]
        
        steady_s_list_sym = [self.Css_sym, self.Kss_sym, self.Lss_sym,
                         self.Iss_sym, self.Yss_sym]     
                         
        steady_s_list_num = [x.subs(self.ss_solutions_dict) for x 
                            in steady_s_list_sym]
                            
        steady_s_vec = np.zeros((6,1))
#        print 'steady_s_vec line 1408', steady_s_vec       

        steady_s_vec[0:-1,:] = np.array([steady_s_list_num]).T
#        print 'steady_s_vec line 1411', steady_s_vec       

        
        if x0 == []:
            x0 = np.zeros( (phi_x.shape[0],1))
        
        x_time_series_mat = np.empty( (x0.shape[0], finalT))
        
#        print '\n'
#        print 'x_time_series_mat.shape', x_time_series_mat.shape
                
        x_non_sta_time_series_mat = np.empty( (x0.shape[0], finalT))
        
        azA_mat = np.empty( (4, finalT))
                
        mua = self.mua
        z_minusone = 0
        z_minusone_hat = 0
        
        current_x = x0
        
        # zt-1, hatzt-1 y hatzt-1enx, are all 0
                
        A_minusone = 1.0
        
        inno_vector = np.zeros((1,1))
        
#        inno_vector[0,0] = self.C[1,1]

#        print 'inno_vector', inno_vector      
        
        w_vector = np.zeros((2,1))
        
        if shock_type=='wa':
            w_vector[0,0] = shock_mag
            
        elif shock_type=='wzetaa':
            w_vector[1,0] = shock_mag
            
            
        if dologs:
            linlinco = np.hstack((phi_x, phi_w))
            vss = self.get_steady_state_x_values()
            sel_vec = np.array([True, True, True, False, True, False,
                                  False, False]) 
            v_allss_xw = np.vstack((vss,np.zeros((3,1))))
            loglog_coeffs = self.from_linlin_to_loglog(linlinco, v_allss_xw,
                                                       sel_vec)
            phi_x = loglog_coeffs[:,0:6]
            phi_w = loglog_coeffs[:,6:7]
            #steady_s_vec = np.vstack((vss,np.zeros((1,1)))) #only x variables
            steady_s_vec[sel_vec] = np.log(steady_s_vec[sel_vec])
            
                   
        a_zero = np.dot(self.D, np.array( (mua, z_minusone))) + np.dot(self.G, w_vector)        
#        
#        print '\n'
#        print 'np.dot(self.G, w_vector)',np.dot(self.G, w_vector)
#        print '\n'
#        print 'a_zero', a_zero
#        print '\n'
#        print 'np.dot(self.D, np.array( (mua, z_minusone_hat)))', np.dot(self.D, np.array( (mua, z_minusone_hat)))
        
       
        inno_zero =  a_zero - np.dot(self.D, np.array( (mua, z_minusone_hat))) 
        
        inno_vector[0,0] = inno_zero 
        
#        print '\n'
#        print 'inno_zero', inno_zero
#        print 'phi_x', phi_x
#        print 'x0', x0
#        print 'np.dot(phi_x, x0)', np.dot(phi_x, x0)
#        print 'phi_w', phi_w
#        print 'inno_vector', inno_vector
#        print 'np.dot(phi_w, inno_vector)', np.dot(phi_w, inno_vector)
#               
#        print 'steady_s_vec line 1485', steady_s_vec       
                
        
        first_response_x = steady_s_vec + np.dot(phi_x, x0) + np.dot(phi_w, inno_vector)
        
        # zt-1inx has been updated, first time
        
#        print '\n'
#        print 'steady_s_vec.shape', steady_s_vec.shape
#        print 'np.dot(phi_x, x0).shape', np.dot(phi_x, x0).shape
#        print 'np.dot(phi_w, inno_vector).shape', np.dot(phi_w, inno_vector).shape
#        print 'first_response_x.shape', first_response_x.shape
        
        
        # onte-time updated zt-1inx is stored in x_time_series
        x_time_series_mat[:,0] = first_response_x[:,0]   
#        x_non_sta_time_series_mat[:,0] = first_response_x[:,0]/A_minusone 
        x_non_sta_time_series_mat[:,0] = first_response_x[:,0]*A_minusone 
        
        
        zetaa_zero = self.rhoa*z_minusone + np.dot(self.C,w_vector)[1,:]
        zetaa_zero_hat = self.rhoa*z_minusone_hat + np.dot(self.KalG_ss, inno_zero)[1,:]
        
        
        # zt-1 and hatzt-1 get their first update        
        
#        
#        print '\n'
#        print 'zetaa_zero',  zetaa_zero      
#        print 'zetaa_zero_hat',  zetaa_zero_hat      
        
        
        A_zero = A_minusone*np.exp(a_zero)        
        
        azA_mat[0,0] = a_zero # a_0 has the direct impact of the impulse
        azA_mat[1,0] = zetaa_zero # z_{0} has the direct impact of the impulse  
        azA_mat[2,0] = zetaa_zero_hat # A_{0} has the direct impact of the impulse
        azA_mat[3,0] = A_minusone # A_{0} has the direct impact of the impulse
        
        current_zetaa = zetaa_zero
        current_zetaa_hat = zetaa_zero_hat
        current_inno = inno_zero
        
        current_A = A_zero
        
        current_x = first_response_x # C_0, K_1, L_0 etc.
        
#        print '\n'
#        print 'before the loop:'
#        print 'current_zetaa: ', current_zetaa
#        print 'current_zetaa_hat: ', current_zetaa_hat
#        print 'current_x[5]', current_x[5]
#        print 'azA_mat[1,0] (i.e. zetaa)', azA_mat[1,0]
#        print 'azA_mat[2,0] (i.e. zetaa_hat)', azA_mat[2,0]
#        print 'azA_mat[2,0]-current_zetaa_hat', azA_mat[2,0]-current_zetaa_hat

        
        for t in range(finalT-1):  
            
#            print '\n'
#            print 'inside the loop:'
#            print 't=',t
#            print 'current_zetaa: ', current_zetaa
#            print 'current_zetaa_hat: ', current_zetaa_hat
#            print 'current_x[5]', current_x[5]
#            print 'azA_mat[1,t] (i.e. zetaa)', azA_mat[1,t]
#            print 'azA_mat[2,t] (i.e. zetaa_hat)', azA_mat[2,t]
#            print 'azA_mat[2,t]-current_zetaa_hat', azA_mat[2,t]-current_zetaa_hat
#
#            
#            print '\n'
#            print 't=', t
#            print 'current_zetaa', current_zetaa            
#            print 'current_zetaa_hat', current_zetaa_hat
#
#            print '\n'
#            print 'current_x.shape', current_x.shape
#                   
            
            current_a = np.dot(self.D, np.array( (mua, current_zetaa))) # those are a_1 and z_0
            current_inno = current_a - np.dot(
                self.D, np.array( (mua, current_zetaa_hat)))
                
            inno_vector[0,0] = current_inno
                
#            print 'np.dot(self.KalG_ss, current_inno)',np.dot(self.KalG_ss, current_inno)    

            current_zetaa_hat = self.rhoa*current_zetaa_hat + \
                                np.dot(self.KalG_ss, current_inno)[1]
            
#            print 'steady_s_vec.shape', steady_s_vec.shape 
#            print 'np.dot(phi_x, current_x).shape', np.dot(phi_x, current_x).shape
#            print 'np.dot(phi_w, current_inno).shape', np.dot(phi_w, inno_vector).shape 
#           
            
            
            
            current_x = steady_s_vec + np.dot(phi_x, current_x-steady_s_vec) + \
                np.dot(phi_w, inno_vector)
       
#            print 'current_x.shape', current_x.shape 
            
            x_time_series_mat[:,t+1] = current_x[:,0] # C_1, K_2, L_1 ...
            # \tilde{C}_1, \tilde{K}_2, L_1 ...
#            x_non_sta_time_series_mat[:,t+1] = current_x[:,0]/current_A 
            if dologs:
                x_non_sta_time_series_mat[:,t+1] = current_x[:,0] +  np.log(current_A) 
            else:
                x_non_sta_time_series_mat[:,t+1] = current_x[:,0]*current_A
                        
            
            current_zetaa = self.rhoa*current_zetaa # z_1 and z_0
            
            azA_mat[0,t+1] = current_a  # a_1, a_2 ...
            azA_mat[1,t+1] = current_zetaa  # z_1, z_2 
            azA_mat[2,t+1] = current_zetaa_hat  # A_0, A_1
            azA_mat[3,t+1] = current_A  # A_0, A_1
            
            current_A = current_A*np.exp(current_a) # A_1, A_0, a_1
        
        x_non_sta_time_series_mat[2,:] = x_time_series_mat[2,:]
        x_non_sta_time_series_mat[5,:] = x_time_series_mat[5,:]
        
        BGP_det = np.zeros(x_non_sta_time_series_mat.shape)
        for i in range(BGP_det.shape[0]):
            BGP_det[i,:] = azA_mat[3,:]*steady_s_vec[i]
            
        

        return azA_mat, x_time_series_mat, x_non_sta_time_series_mat, BGP_det


class RobustPartialInfoModel(PartialInfoModel):
    x = 0

    def __init__(self, statespace,  alpha = 0.33, delta = 0.1, gamma=1,
                 beta = 0.95, rhoa=0.95, nu=0.5, muatrue = 0.03, 
                 modeltype = 'toy', statespace_symdict = {}, ss_x_dict={},
                 xvarstring='CKL', prior_mean=[], prior_variance=[],
                 theta=10):
                     
        PartialInfoModel.__init__(self, statespace, alpha, delta, gamma, beta,
                                  rhoa, nu, muatrue, modeltype,
                                  statespace_symdict,ss_x_dict, xvarstring, prior_mean,
                                  prior_variance)
                         
        self.theta = theta
        self.add_model_symbols_rpi()
        
        self.q = self.modelsymbols_dict['q'] 
                
        
    def add_model_symbols_rpi(self):
        
        q = sympy.Symbol('q', positive=True)
        
        additional_symbols_dict = {'q':q}            
        
        self.modelsymbols_dict.update(additional_symbols_dict)
        

        
    def gvec_toymodel_rpi(self, symbolic_elem = 'just_x_w'):
        r"""
        Return a list with the LHS of 
        $g(x_tp1,x_t,x_tm1,w_tp1,w_t)=0$, describing the equilibrium
        conditions of the model and using user-supplied parameter values.
        Only variables in $x$ and $w$ remain as symbols. 
        The main goal is to provide a way to evaluate approximation errors and
        a sympy expression to differentiate w.r.t $x$ and $w$
        """        
        

        if symbolic_elem=='all_but_volatilities':
            alpha, beta = self.alpha_sym, self.beta_sym
            delta, nu = self.delta_sym, self.nu_sym
            A, D = self.A_sym, self.D_sym
            C, G =  self.C, self.G
            KalG_ss = self.KalG_ss
       
            
        elif symbolic_elem=='just_x_w':
            alpha, beta, delta, nu = self.alpha, self.beta, self.delta, self.nu
            A, C, D, G = self.A, self.C, self.D, self.G
            KalG_ss = self.KalG_ss
            
 
        elif symbolic_elem=='all':
            alpha, beta = self.alpha_sym, self.beta_sym
            delta, nu = self.delta_sym, self.nu_sym
            A, C, D, G = self.A_sym, self.C_sym, self.D_sym, self.G_sym
            KalG_ss = self.KalG_ss_sym
        
       
        
        zt_sym_hat, ztm1_sym_hat = self.zt_sym_hat, self.ztm1_sym_hat
        zetaat_sym_hat = self.zetaat_sym_hat
        
        innot_sym, innotp1_sym = self.innot_sym, self.innotp1_sym
        
        q = self.q
        
        
        Ct, Lt, Kstate =  self.Ct_sym, self.Lt_sym, self.Kstate_sym 
        Kpolicy, Ctp1, Ltp1 =  self.Kpolicy_sym, self.Ctp1_sym, self.Ltp1_sym 
       
       
       
        at = (D*ztm1_sym_hat + q*innot_sym)[0]
        atp1 = (D *zt_sym_hat + q*innotp1_sym)[0]
        
        if self.xvarstring=='CKL':
            # at, atp1, Yt are not primitive objects, but formulas are much shorter
            # if using these intermediate expressions:
            
        
            Yt = sympy.exp((1-alpha)*at)  * Kstate**(alpha) * Lt**(1-alpha) 
            Ytp1 = sympy.exp((1-alpha)*atp1)  * Kpolicy**(alpha) * Ltp1**(1-alpha) 
        
            # Toy model FOCs, under full information
            g1 = beta*sympy.exp(-at) * (Ct/Ctp1) * \
                (alpha * Ytp1/Kpolicy    +(1-delta)) - 1
        
            g2 = Yt - sympy.exp(at) * Kpolicy + (1-delta)*Kstate - Ct 
        
            g3 = nu*(1-alpha)*Yt/(Ct*(1-nu)) - Lt/(1-Lt)
        
            # state space of productivity growth are also equilibrium conditions
            g4 = zetaat_sym_hat - (A*ztm1_sym_hat + KalG_ss*q*innot_sym)[1]
            
            glist = [g1, g2, g3, g4]
        
        elif self.xvarstring=='CKLIY':
            # Toy model FOCs, under full information
        
            Yt = self.Yt_sym
            Ytp1 = self.Ytp1_sym
            It = self.It_sym         
            
            g1 = beta*sympy.exp(-at) * (Ct/Ctp1) * \
                (alpha * Ytp1/Kpolicy    +(1-delta)) - 1
        
            g2 = Yt - It - Ct 
        
            g3 = (nu/(1-nu))*(1-alpha)*Yt/Ct - Lt/(1-Lt)
            
            g4 = Yt - sympy.exp((1-alpha)*at)  * Kstate**(alpha) * Lt**(1-alpha)
            
            g5 = It - sympy.exp(at) * Kpolicy + (1-delta)*Kstate
        
            # state space of productivity growth are also equilibrium conditions
            g6 = zetaat_sym_hat - (A*ztm1_sym_hat + KalG_ss*q*innot_sym)[1]
            
            glist = [g1, g2, g3, g4, g5, g6]
                
        
        if symbolic_elem=='just_x_w':
            glist = [g.subs({self.mua_sym: self.mua, 
                              self.rhoa_sym: self.rhoa}) for g in glist]
        
#        gvec = (g1, g2, g3, g4)       
        gvec = sympy.Matrix( glist )       
        
        return gvec

    def get_gq_at_ss(self, symbolic_elem='just_x_w', 
                                        subs_x_to_ss_num=True):
        
        sym_choice = symbolic_elem
        
        gvec = self.gvec_toymodel_fi(symbolic_elem = sym_choice)
        
        gvec_d_q = gvec.jacobian([self.q])
        
        gvec_d_q_at_ss = self.make_expressions_at_ss_fi(gvec_d_q)
        
        
        if subs_x_to_ss_num==True:
            if self.ss_solutions_dict == {}:
                self.update_ss_solutions_fi_toy()
            gvec_d_q_at_ss = gvec_d_q_at_ss.subs(self.ss_solutions_dict)
            gvec_d_q_at_ss_q0 = gvec_d_q_at_ss.subs({'q':0})
            
            gvec_d_q_at_ss = self.matrix2numpyfloat(gvec_d_q_at_ss)
            gvec_d_q_at_ss_q0 = self.matrix2numpyfloat(gvec_d_q_at_ss_q0)
         
        
        return  gvec_d_q_at_ss_q0, gvec_d_q_at_ss  
                
    def get_first_ord_coef_rpi(self, isPartialInfo=True, isloglog=False):
        gxm1ss_pi, gxss_pi, gxp1ss_pi, gwss_pi, gwp1ss_pi = self.gvec_diffs_ev_at_ss_toymodel_pi()
        
        pistatus = isPartialInfo
        
        phi_x, phi_w = self.get_first_order_approx_coeff(isPartialInfo=pistatus)
                                 
        gq, gq_unevq = self.get_gq_at_ss()
                                 
        phi_q_inv_term = scipy.linalg.inv(np.dot(gxp1ss_pi, phi_x) + \
                        gxp1ss_pi + gxss_pi)
                        
        coef_on_dist_E = np.dot(gxp1ss_pi, phi_w) + gwp1ss_pi
        
        utility_ss_sym = self.nu * sympy.log(self.Ct_sym) + \
                        (1-self.nu) * sympy.log(self.Lt_sym)
        utility_ss_sym = sympy.Matrix([utility_ss_sym])        
                
        du_dx =  utility_ss_sym.jacobian([self.xvar_t_sym])   
        du_dx = self.make_expressions_at_ss_fi(du_dx)
        du_dx_at_ss = du_dx.subs(self.ss_solutions_dict)
        du_dx_at_ss = self.matrix2numpyfloat(du_dx_at_ss)
        nx = len(self.xvar_t_sym)
        Ibetaphi = np.eye(nx) - self.beta*phi_x
        invIbetaphi = scipy.linalg.inv(Ibetaphi)
        Vx = np.dot(du_dx_at_ss, invIbetaphi)
        
        # we need Vx for every date, evaluating dVdx not at SS but
        # at each x in the impulse-response simulation        
        
        E_w_dist = - np.dot(Vx,phi_w).T/self.theta
        
        vec_for_diag = np.dot(coef_on_dist_E, E_w_dist)
        
#        print 'E_w_dist', E_w_dist
#        print 'coef_on_dist_E', coef_on_dist_E
#        print 'vec_for_diag', vec_for_diag
#        
        diag_mat = np.diag(vec_for_diag)
        
        phi_q = - np.dot(phi_q_inv_term, gq+diag_mat)
        
#        print 'phi_q', phi_q

        if isloglog == True:
            linlinco = np.hstack((phi_x, phi_w, phi_q))
#            print 'phi_x.shape',phi_x.shape
#            print 'phi_w.shape',phi_w.shape
#            print 'phi_q.shape',phi_q.shape
            
            vss = self.get_steady_state_x_values()
            sel_vec = np.array([True, True, True, False, True, False,
                                  False, False, False]) 
            v_allss_xwq = np.vstack((vss,np.zeros((4,1))))
#            print 'linlinco.shape',linlinco.shape
            loglog_coeffs = self.from_linlin_to_loglog(linlinco, v_allss_xwq,
                                                       sel_vec)
#            print 'loglog_coeffs.shape',loglog_coeffs.shape
            phi_x = loglog_coeffs[:,0:6]
            phi_w = loglog_coeffs[:,6:7]
            phi_q = loglog_coeffs[:,7]
                                 
        return phi_x, phi_w, phi_q                         
#        return phi_x.T, phi_w.T, phi_q.T                

    def impulse_response_toy_rpi(self, phi_x, phi_w, phi_q, shock = 1, x0 = [],
                                finalT = 30, shock_type='wa', shock_mag=1,
                                dologs=False):
        
        azA, xsta, xmat, BGP = self.impulse_response_toy_pi(phi_x, phi_w,
                                shock_type, finalT,
                                shock_mag, dologs)
        
        xsta_r = xsta + phi_q
        
        return azA, xsta, xmat, BGP, xsta_r
                                
                                
                                
                                
        



class RobustFullInfoModel(FullInfoModel):
    x = 0

    def __init__(self, statespace,  alpha = 0.33, delta = 0.1, gamma=1,
                 beta = 0.95, rhoa=0.95, nu=0.5, muatrue = 0.03, 
                 modeltype = 'toy', statespace_symdict = {}, ss_x_dict={},
                 xvarstring='CKL',
                 theta=100):
                     
        FullInfoModel.__init__(self, statespace, alpha, delta, gamma, beta,
                                  rhoa, nu, muatrue, modeltype,
                                  statespace_symdict, ss_x_dict, xvarstring)
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                         
        self.theta = theta
        self.add_model_symbols_rfi()
        
        self.q = self.modelsymbols_dict['q'] 
                
        
    def add_model_symbols_rfi(self):
        
        q = sympy.Symbol('q', positive=True)
        
        additional_symbols_dict = {'q':q}            
        
        self.modelsymbols_dict.update(additional_symbols_dict)
        

        
    def gvec_toymodel_rfi(self, symbolic_elem = 'just_x_w'):
        r"""
        Return a list with the LHS of 
        $g(x_tp1,x_t,x_tm1,w_tp1,w_t)=0$, describing the equilibrium
        conditions of the model and using user-supplied parameter values.
        Only variables in $x$ and $w$ remain as symbols. 
        The main goal is to provide a way to evaluate approximation errors and
        a sympy expression to differentiate w.r.t $x$ and $w$
        """        
        

        if symbolic_elem=='all_but_volatilities':
            alpha, beta = self.alpha_sym, self.beta_sym
            delta, nu = self.delta_sym, self.nu_sym
            A, D = self.A_sym, self.D_sym
            C, G =  self.C, self.G
            KalG_ss = self.KalG_ss
       
            
        elif symbolic_elem=='just_x_w':
            alpha, beta, delta, nu = self.alpha, self.beta, self.delta, self.nu
            A, C, D, G = self.A, self.C, self.D, self.G
            KalG_ss = self.KalG_ss
            
 
        elif symbolic_elem=='all':
            alpha, beta = self.alpha_sym, self.beta_sym
            delta, nu = self.delta_sym, self.nu_sym
            A, C, D, G = self.A_sym, self.C_sym, self.D_sym, self.G_sym
            KalG_ss = self.KalG_ss_sym
        
       
        
        zt_sym_hat, ztm1_sym_hat = self.zt_sym_hat, self.ztm1_sym_hat
        zetaat_sym_hat = self.zetaat_sym_hat
        
        innot_sym, innotp1_sym = self.innot_sym, self.innotp1_sym
        
        q = self.q
        
        
        Ct, Lt, Kstate =  self.Ct_sym, self.Lt_sym, self.Kstate_sym 
        Kpolicy, Ctp1, Ltp1 =  self.Kpolicy_sym, self.Ctp1_sym, self.Ltp1_sym 
       
       
       
        at = (D*ztm1_sym_hat + q*innot_sym)[0]
        atp1 = (D *zt_sym_hat + q*innotp1_sym)[0]
        
        if self.xvarstring=='CKL':
            # at, atp1, Yt are not primitive objects, but formulas are much shorter
            # if using these intermediate expressions:
            
        
            Yt = sympy.exp((1-alpha)*at)  * Kstate**(alpha) * Lt**(1-alpha) 
            Ytp1 = sympy.exp((1-alpha)*atp1)  * Kpolicy**(alpha) * Ltp1**(1-alpha) 
        
            # Toy model FOCs, under full information
            g1 = beta*sympy.exp(-at) * (Ct/Ctp1) * \
                (alpha * Ytp1/Kpolicy    +(1-delta)) - 1
        
            g2 = Yt - sympy.exp(at) * Kpolicy + (1-delta)*Kstate - Ct 
        
            g3 = nu*(1-alpha)*Yt/(Ct*(1-nu)) - Lt/(1-Lt)
        
            # state space of productivity growth are also equilibrium conditions
            g4 = zetaat_sym_hat - (A*ztm1_sym_hat + KalG_ss*q*innot_sym)[1]
            
            glist = [g1, g2, g3, g4]
        
        elif self.xvarstring=='CKLIY':
            # Toy model FOCs, under full information
        
            Yt = self.Yt_sym
            Ytp1 = self.Ytp1_sym
            It = self.It_sym         
            
            g1 = beta*sympy.exp(-at) * (Ct/Ctp1) * \
                (alpha * Ytp1/Kpolicy    +(1-delta)) - 1
        
            g2 = Yt - It - Ct 
        
            g3 = (nu/(1-nu))*(1-alpha)*Yt/Ct - Lt/(1-Lt)
            
            g4 = Yt - sympy.exp((1-alpha)*at)  * Kstate**(alpha) * Lt**(1-alpha)
            
            g5 = It - sympy.exp(at) * Kpolicy + (1-delta)*Kstate
        
            # state space of productivity growth are also equilibrium conditions
            g6 = zetaat_sym_hat - (A*ztm1_sym_hat + KalG_ss*q*innot_sym)[1]
            
            glist = [g1, g2, g3, g4, g5, g6]
                
        
        if symbolic_elem=='just_x_w':
            glist = [g.subs({self.mua_sym: self.mua, 
                              self.rhoa_sym: self.rhoa}) for g in glist]
        
#        gvec = (g1, g2, g3, g4)       
        gvec = sympy.Matrix( glist )       
        
        return gvec

    def get_gq_at_ss(self, symbolic_elem='just_x_w', 
                                        subs_x_to_ss_num=True):
        
        sym_choice = symbolic_elem
        
        gvec = self.gvec_toymodel_fi(symbolic_elem = sym_choice)
        
        gvec_d_q = gvec.jacobian([self.q])
        
        gvec_d_q_at_ss = self.make_expressions_at_ss_fi(gvec_d_q)
        
        
        if subs_x_to_ss_num==True:
            if self.ss_solutions_dict == {}:
                self.update_ss_solutions_fi_toy()
            gvec_d_q_at_ss = gvec_d_q_at_ss.subs(self.ss_solutions_dict)
            gvec_d_q_at_ss_q0 = gvec_d_q_at_ss.subs({'q':0})
            
            gvec_d_q_at_ss = self.matrix2numpyfloat(gvec_d_q_at_ss)
            gvec_d_q_at_ss_q0 = self.matrix2numpyfloat(gvec_d_q_at_ss_q0)
         
        
        return  gvec_d_q_at_ss_q0, gvec_d_q_at_ss  
                
    def get_first_ord_coef_rfi(self, isPartialInfo=False, isloglog=False):
        gxm1ss_pi, gxss_pi, gxp1ss_pi, gwss_pi, gwp1ss_pi = self.gvec_diffs_ev_at_ss_toymodel_fi()
        
        pistatus = isPartialInfo
        
        phi_x, phi_w = self.get_first_order_approx_coeff(isPartialInfo=pistatus)
                                 
        gq, gq_unevq = self.get_gq_at_ss()
                                 
        phi_q_inv_term = scipy.linalg.inv(np.dot(gxp1ss_pi, phi_x) + \
                        gxp1ss_pi + gxss_pi)
                        
        coef_on_dist_E = np.dot(gxp1ss_pi, phi_w) + gwp1ss_pi
        
        utility_ss_sym = self.nu * sympy.log(self.Ct_sym) + \
                        (1-self.nu) * sympy.log(self.Lt_sym)
        utility_ss_sym = sympy.Matrix([utility_ss_sym])        
                
        du_dx =  utility_ss_sym.jacobian([self.xvar_t_sym])   
        du_dx = self.make_expressions_at_ss_fi(du_dx)
        du_dx_at_ss = du_dx.subs(self.ss_solutions_dict)
        du_dx_at_ss = self.matrix2numpyfloat(du_dx_at_ss)
        nx = len(self.xvar_t_sym)
        Ibetaphi = np.eye(nx) - self.beta*phi_x
        invIbetaphi = scipy.linalg.inv(Ibetaphi)
        Vx = np.dot(du_dx_at_ss, invIbetaphi)
        
        E_w_dist = - np.dot(Vx,phi_w).T/self.theta
        
        vec_for_diag = np.dot(coef_on_dist_E, E_w_dist)
        
#        print 'E_w_dist', E_w_dist
#        print 'coef_on_dist_E', coef_on_dist_E
#        print 'vec_for_diag', vec_for_diag
#        
        diag_mat = np.diag(vec_for_diag)
        
        phi_q = - np.dot(phi_q_inv_term, gq+diag_mat)
        
        if isloglog == True:
            linlinco = np.hstack((phi_x, phi_w, phi_q))
#            print 'phi_x.shape',phi_x.shape
#            print 'phi_w.shape',phi_w.shape
#            print 'phi_q.shape',phi_q.shape
            
            vss = self.get_steady_state_x_values()
            sel_vec = np.array([True, True, True, False, True, False,
                                  False, False, False]) 
            v_allss_xwq = np.vstack((vss,np.zeros((4,1))))
#            print 'linlinco.shape',linlinco.shape
            loglog_coeffs = self.from_linlin_to_loglog(linlinco, v_allss_xwq,
                                                       sel_vec)
#            print 'loglog_coeffs.shape',loglog_coeffs.shape
            phi_x = loglog_coeffs[:,0:6]
            phi_w = loglog_coeffs[:,6:8]
            phi_q = loglog_coeffs[:,8]

        
                                 
        return phi_x, phi_w, phi_q                         
#        return phi_x.T, phi_w.T, phi_q.T                         
    

    def impulse_response_toy_rpi(self, phi_x, phi_w, phi_q, shock = 1, x0 = [],
                                finalT = 30, shock_type='wa', shock_mag=1,
                                dologs=False, theta=100):

        
        steady_s_list_sym = [self.Css_sym, self.Kss_sym, self.Lss_sym,
                         self.Iss_sym, self.Yss_sym]     
                         
        steady_s_list_num = [x.subs(self.ss_solutions_dict) for x 
                            in steady_s_list_sym]
                            
        steady_s_vec = np.zeros((6,1))
        steady_s_vec[0:-1,:] = np.array([steady_s_list_num]).T
        
        if x0 == []:
            x0 = np.zeros( (phi_x.shape[0],1))
        
        x_time_series_mat = np.empty( (x0.shape[0], finalT))
                
        x_non_sta_time_series_mat = np.empty( (x0.shape[0], finalT))
        
        azA_mat = np.empty( (4, finalT))
                
        mua = self.mua
        z_minusone = 0
        z_minusone_hat = 0
        
        current_x = x0
        
        # zt-1, hatzt-1 y hatzt-1enx, are all 0
                
        A_minusone = 1.0
        
        inno_vector = np.zeros((1,1))
        
        
        w_vector = np.zeros((2,1))
        
        if shock_type=='wa':
            w_vector[0,0] = shock_mag
            
        elif shock_type=='wzetaa':
            w_vector[1,0] = shock_mag
            
            
        if dologs:
            linlinco = np.hstack((phi_x, phi_w))
            vss = self.get_steady_state_x_values()
            sel_vec = np.array([True, True, True, False, True, False,
                                  False, False]) 
            v_allss_xw = np.vstack((vss,np.zeros((3,1))))
            loglog_coeffs = self.from_linlin_to_loglog(linlinco, v_allss_xw,
                                                       sel_vec)
            phi_x = loglog_coeffs[:,0:6]
            phi_w = loglog_coeffs[:,6:7]
            steady_s_vec = np.vstack((vss,np.zeros((1,1)))) #only x variables
            steady_s_vec[sel_vec] = np.log(steady_s_vec[sel_vec])
            
                   
        a_zero = np.dot(self.D, np.array( (mua, z_minusone))) + np.dot(self.G, w_vector)        
       
        inno_zero =  a_zero - np.dot(self.D, np.array( (mua, z_minusone_hat))) 
        
        inno_vector[0,0] = inno_zero 
        
        first_response_x = steady_s_vec + np.dot(phi_x, x0) + np.dot(phi_w, inno_vector)
        
        # onte-time updated zt-1inx is stored in x_time_series
        x_time_series_mat[:,0] = first_response_x[:,0]   
#        x_non_sta_time_series_mat[:,0] = first_response_x[:,0]/A_minusone 
        x_non_sta_time_series_mat[:,0] = first_response_x[:,0]*A_minusone 
        
        
        zetaa_zero = self.rhoa*z_minusone + np.dot(self.C,w_vector)[1,:]
        zetaa_zero_hat = self.rhoa*z_minusone_hat + np.dot(self.KalG_ss, inno_zero)[1,:]
        
        
        A_zero = A_minusone*np.exp(a_zero)        
        
        azA_mat[0,0] = a_zero # a_0 has the direct impact of the impulse
        azA_mat[1,0] = zetaa_zero # z_{0} has the direct impact of the impulse  
        azA_mat[2,0] = zetaa_zero_hat # A_{0} has the direct impact of the impulse
        azA_mat[3,0] = A_minusone # A_{0} has the direct impact of the impulse
        
        current_zetaa = zetaa_zero
        current_zetaa_hat = zetaa_zero_hat
        current_inno = inno_zero
        
        current_A = A_zero
        
        current_x = first_response_x # C_0, K_1, L_0 etc.
        
        for t in range(finalT-1):  
            
            
            current_a = np.dot(self.D, np.array( (mua, current_zetaa))) # those are a_1 and z_0
            current_inno = current_a - np.dot(
                self.D, np.array( (mua, current_zetaa_hat)))
                
            inno_vector[0,0] = current_inno
                

            current_zetaa_hat = self.rhoa*current_zetaa_hat + \
                                np.dot(self.KalG_ss, current_inno)[1]
            
            
            current_x = steady_s_vec + np.dot(phi_x, current_x-steady_s_vec) + \
                np.dot(phi_w, inno_vector)
           
            x_time_series_mat[:,t+1] = current_x[:,0] # C_1, K_2, L_1 ...
            # \tilde{C}_1, \tilde{K}_2, L_1 ...
#            x_non_sta_time_series_mat[:,t+1] = current_x[:,0]/current_A 
            if dologs:
                x_non_sta_time_series_mat[:,t+1] = current_x[:,0] +  np.log(current_A) 
            else:
                x_non_sta_time_series_mat[:,t+1] = current_x[:,0]*current_A
                        
            
            current_zetaa = self.rhoa*current_zetaa # z_1 and z_0
            
            azA_mat[0,t+1] = current_a  # a_1, a_2 ...
            azA_mat[1,t+1] = current_zetaa  # z_1, z_2 
            azA_mat[2,t+1] = current_zetaa_hat  # A_0, A_1
            azA_mat[3,t+1] = current_A  # A_0, A_1
            
            current_A = current_A*np.exp(current_a) # A_1, A_0, a_1
        
        x_non_sta_time_series_mat[2,:] = x_time_series_mat[2,:]
        x_non_sta_time_series_mat[5,:] = x_time_series_mat[5,:]
        
        BGP_det = np.zeros(x_non_sta_time_series_mat.shape)
        for i in range(BGP_det.shape[0]):
            BGP_det[i,:] = azA_mat[3,:]*steady_s_vec[i]
            
        

        return azA_mat, x_time_series_mat, x_non_sta_time_series_mat, BGP_det


    



#####################################################################
#alphavalue = 0.36
#betavalue = 0.98
#deltavalue = 0.1
#nuvalue = 0.29
#truemuavalueToy = 0.03
#rhoavalue = 0.97 # 0.01087417, 0.00213658
#sigmaza = 0.01*1
#sigmaa = 0.01*7.5
##sigmaa = 0.01
#muavalueSSFI = truemuavalueToy
#muatm = truemuavalueToy

def make_model_instances(alphavalue=0.36, betavalue=0.98, deltavalue=0.1,
                         truemuavalue = 0.03, nuvalue = 0.29, rhoavalue=0.97,
                         sigmaa=0.01*7.5, sigmaza=0.01, theta=100):
                             
#    muavalueSSFI = truemuavalue
#    muatm = truemuavalue
    Aelw =  np.array([[1,0],[0, rhoavalue]])
    Celw = np.array([[0,0],[0,sigmaza]])
    Delw = np.array([[1.0, rhoavalue]])
    Gelw = np.array([[sigmaa, sigmaza]])
    
    
    rhoa_sym = sympy.Symbol('rho_a', positive=True)
    Ds_elw = sympy.Matrix([[1, rhoa_sym]])
    stsp_sdict = {'D':Ds_elw} 
    tech_dict = {'alpha':alphavalue, 'delta':deltavalue}
    prefs_dict = {'beta':betavalue, 'nu':nuvalue}
    prod_dict = {'rhoa':rhoavalue, 'muatrue':truemuavalue}
    
    tech_prefs_prod_dict = {}
    tech_prefs_prod_dict.update(tech_dict)
    tech_prefs_prod_dict.update(prefs_dict)
    tech_prefs_prod_dict.update(prod_dict)
                           
    ss_elw = StateSpace(A=Aelw, C=Celw, D=Delw, G=Gelw)
    
    z_initial = np.zeros((2,1))
    z_initial[0,0] = truemuavalue
    Sigma_initial = 0.01*np.eye(2)
    
    elw_fi = FullInfoModel(ss_elw, statespace_symdict=stsp_sdict,
                       xvarstring='CKLIY', **tech_prefs_prod_dict)  
                       
    ss_dict = elw_fi.get_steady_state_x_dict()

    elw_pi = PartialInfoModel(ss_elw, statespace_symdict=stsp_sdict,
                           xvarstring='CKLIY',prior_mean=z_initial, 
                            ss_x_dict = ss_dict,
                       prior_variance=Sigma_initial, **tech_prefs_prod_dict) 
                           
    elw_rpi = RobustPartialInfoModel(ss_elw, statespace_symdict=stsp_sdict,
                           xvarstring='CKLIY', prior_mean=z_initial, 
                           ss_x_dict = ss_dict,
                       prior_variance=Sigma_initial, theta=theta,
                           **tech_prefs_prod_dict)     
                           
    elw_rfi = RobustFullInfoModel(ss_elw, statespace_symdict=stsp_sdict,
                                  ss_x_dict = ss_dict,
                           xvarstring='CKLIY', theta=theta,
                           **tech_prefs_prod_dict)  
                           
    return elw_fi, elw_pi, elw_rfi, elw_rpi


def make_logs_from_lin(linlinco, ckliy_ss_vals, selec_vec=[], isPartial=False,
                       isRobust=False):
    
    ckliyss = ckliy_ss_vals
    
    if selec_vec == []:
        selec_vec = np.array([True, True, True, False, True, False, False, False]) 
                                  
    if isPartial:
        if isRobust:
            v_allss = np.vstack((ckliyss,np.zeros((3,1))))
        else:
            v_allss = np.vstack((ckliyss,np.zeros((2,1))))
    else:
        if isRobust:
            v_allss = np.vstack((ckliyss,np.zeros((4,1))))
        else:
            v_allss = np.vstack((ckliyss,np.zeros((3,1))))
            
    rowx = linlinco.shape[0]
    colx = linlinco.shape[1]
    
    ss_vec_x = v_allss[0:6]
    
    loglog_coeffs = np.empty(linlinco.shape)
    
    for i in range(rowx):
        for j in range(colx):
            if selec_vec[i]:
                valdenom = ss_vec_x[i]
            else:
                valdenom = 1.0
            if selec_vec[j]:
                valnum = v_allss[j]
            else:
                valnum = 1.0
            ratio_ss = valnum/valdenom 
            loglog_coeffs[i,j] = ratio_ss*linlinco[i,j]    
    
    
    phi_x = loglog_coeffs[:,0:6]
    if isPartial==True:
        phi_w = loglog_coeffs[:,6:7]
        if isRobust:
            phi_q = loglog_coeffs[:,7]
            return phi_x, phi_w, phi_q
        else:
            return phi_x, phi_w
    else:
        phi_w = loglog_coeffs[:,6:8]
        if isRobust:
            phi_q = loglog_coeffs[:,8]
            return phi_x, phi_w, phi_q
        else:
            return phi_x, phi_w
            
            
def set_get_psi_toy(alpha = 0.33, delta = 0.1, gamma=1,
                 beta = 0.95, rhoa=0.95, nu=0.5, muatrue = 0.03,
                 sigmaza = 0.01*1, sigmaa = 0.01*7.5,info_types = ["FullInfo"],
                 prior_mean=[], prior_variance=[], theta=100, dologs=False):
                     
    Aelw =  np.array([[1,0],[0, rhoa]])
    Celw = np.array([[0,0],[0,sigmaza]])
    Delw = np.array([[1.0, rhoa]])
    Gelw = np.array([[sigmaa, sigmaza]])
    
    
    rhoa_sym = sympy.Symbol('rho_a', positive=True)
    Ds_elw = sympy.Matrix([[1, rhoa_sym]])
    stsp_sdict = {'D':Ds_elw} 
    tech_dict = {'alpha':alpha, 'delta':delta}
    prefs_dict = {'beta':beta, 'nu':nu}
    prod_dict = {'rhoa':rhoa, 'muatrue':muatrue}
    
    tech_prefs_prod_dict = {}
    tech_prefs_prod_dict.update(tech_dict)
    tech_prefs_prod_dict.update(prefs_dict)
    tech_prefs_prod_dict.update(prod_dict)
    
    z_dict = {"prior_mean":prior_mean, "prior_variance":prior_variance}
    tech_prefs_prod_z_dict = tech_prefs_prod_dict.copy()
    tech_prefs_prod_z_dict.update(z_dict)
    
    ss_elw = StateSpace(A=Aelw, C=Celw, D=Delw, G=Gelw)
    
#    toy_theta=100
    
    coeffs_lists = []
    
    # create FullInfo, because is the cheapest way to get steady state ckliy
    elw_fi = FullInfoModel(ss_elw, statespace_symdict=stsp_sdict,
                       xvarstring='CKLIY', **tech_prefs_prod_dict)  
                       
    ckliy_ss_dict = elw_fi.get_steady_state_x_dict()
    
    if 'FullInfo' in info_types:
        psi_x_fi, psi_w_fi = elw_fi.get_first_order_approx_coeff(
                                isloglog=dologs)
        coeffs_lists.append((psi_x_fi, psi_w_fi))
        
    if 'PartialInfo' in info_types:
        
        elw_pi = PartialInfoModel(ss_elw, statespace_symdict=stsp_sdict,
                           xvarstring='CKLIY', ss_x_dict = ckliy_ss_dict,
                               **tech_prefs_prod_z_dict)
        
        psi_x_pi, psi_w_pi = elw_pi.get_first_order_approx_coeff(
                                                    isPartialInfo=True,
                                                    isloglog=dologs)
                                                    
        coeffs_lists.append((psi_x_pi, psi_w_pi))
                                                    
    if 'RobustFullInfo' in info_types:
        
        elw_rfi = RobustFullInfoModel(ss_elw, statespace_symdict=stsp_sdict,
                           xvarstring='CKLIY', ss_x_dict = ckliy_ss_dict,
                           theta=theta, **tech_prefs_prod_dict)
        
        psi_x_rfi, psi_w_rfi, psi_q_rfi = elw_rfi.get_first_ord_coef_rfi(
                            isloglog=dologs)
                                                    
        coeffs_lists.append((psi_x_rfi, psi_w_rfi, psi_q_rfi))


    if 'RobustPartialInfo' in info_types:
        
        elw_rpi = RobustPartialInfoModel(ss_elw, statespace_symdict=stsp_sdict,
                           xvarstring='CKLIY', ss_x_dict = ckliy_ss_dict,
                             theta=theta, **tech_prefs_prod_z_dict)
        
        psi_x_rpi, psi_w_rpi, psi_q_rpi = elw_rpi.get_first_ord_coef_rpi(
        isPartialInfo=True, isloglog=dologs)
                                                    
        coeffs_lists.append((psi_x_rpi, psi_w_rpi, psi_q_rpi))
                
    return coeffs_lists
        
        
        
def par_vs_par(set_of_par_x=[], par_name_x="", par_tex_x="", set_of_par_lev=[],
               par_name_lev="", par_tex_lev="", baseline_dict={},
               set_of_info_types=['FullInfo'], postname="", dologs=True): 
                   
    
    changed_dict = baseline_dict.copy()
    num_of_par_x = len(set_of_par_x)    
#    num_infos = len(set_of_info_types)
    info_count = 0
    nxpar = len(set_of_par_x)
    nlevpar = len(set_of_par_lev)    
    nvars = 5           
    var_xpar_levpar = np.empty((nvars,nlevpar,nxpar))         
    if "RobustFullInfo" in set_of_info_types  or "RobustPartialInfo" in set_of_info_types :
        var_xpar_levpar_q = np.empty((nvars,nlevpar,nxpar))  
        

    for info in set_of_info_types:
        
        psi_wza_con_change_par_x = np.empty((num_of_par_x,1))
        psi_wza_kap_change_par_x = np.empty((num_of_par_x,1))
        psi_wza_lab_change_par_x = np.empty((num_of_par_x,1))
        psi_wza_inv_change_par_x = np.empty((num_of_par_x,1))
        psi_wza_out_change_par_x = np.empty((num_of_par_x,1))
        
        fig_con_parx_parlev, ax_con_parx_parlev = plt.subplots()
        fig_kap_parx_parlev, ax_kap_parx_parlev = plt.subplots()
        fig_lab_parx_parlev, ax_lab_parx_parlev = plt.subplots()
        fig_inv_parx_parlev, ax_inv_parx_parlev = plt.subplots()
        fig_out_parx_parlev, ax_out_parx_parlev = plt.subplots()
           
        
        if info=="RobustFullInfo" or info=="RobustPartialInfo":
            fig_q_con_parx_parlev, ax_q_con_parx_parlev = plt.subplots()
            fig_q_kap_parx_parlev, ax_q_kap_parx_parlev = plt.subplots()
            fig_q_lab_parx_parlev, ax_q_lab_parx_parlev = plt.subplots()
            fig_q_inv_parx_parlev, ax_q_inv_parx_parlev = plt.subplots()
            fig_q_out_parx_parlev, ax_q_out_parx_parlev = plt.subplots()
            
            psi_q_con_change_par_x = np.empty((num_of_par_x,1))
            psi_q_kap_change_par_x = np.empty((num_of_par_x,1))
            psi_q_lab_change_par_x = np.empty((num_of_par_x,1))
            psi_q_inv_change_par_x = np.empty((num_of_par_x,1))
            psi_q_out_change_par_x = np.empty((num_of_par_x,1))
            
        lev_par_count = 0
        for this_par_lev in set_of_par_lev:
            changed_dict[par_name_lev] = this_par_lev  
            print this_par_lev
            count = 0
            for this_par_x in set_of_par_x:
                changed_dict[par_name_x] = this_par_x
            
                this_psi_all = set_get_psi_toy(info_types=[info],
                                               dologs=dologs, **changed_dict)
                                        
                this_psi = this_psi_all[info_count]
                
                this_psi_x, this_psi_w = this_psi[0], this_psi[1]
                                                    
                
                if info=="FullInfo" or info=="RobustFullInfo":
                    psi_wza_con =  this_psi_w[0,1]
                    psi_wza_kap =  this_psi_w[1,1]
                    psi_wza_lab =  this_psi_w[2,1]
                    psi_wza_inv =  this_psi_w[3,1]
                    psi_wza_out =  this_psi_w[4,1]
                    
                    w_name = "wza"
                    psiw_name = "$\\psi_{x,wza}$"
                    if info=="RobustFullInfo":
                        this_psi_q = this_psi[2]
                        psi_q_con =  this_psi_q[0]
                        psi_q_kap =  this_psi_q[1]
                        psi_q_lab =  this_psi_q[2]
                        psi_q_inv =  this_psi_q[3]
                        psi_q_out =  this_psi_q[4]
                        
                else:
                    psi_wza_con =  this_psi_w[0]
                    psi_wza_kap =  this_psi_w[1]
                    psi_wza_lab =  this_psi_w[2]
                    psi_wza_inv =  this_psi_w[3]
                    psi_wza_out =  this_psi_w[4]
                    w_name = "w"
                    psiw_name = "$\\psi_{x,w}$"
                    if info=="RobustPartialInfo":
                        this_psi_q = this_psi[2]
#                        print "\nthis_psi_q: ", this_psi_q
                        psi_q_con =  this_psi_q[0]
                        psi_q_kap =  this_psi_q[1]
                        psi_q_lab =  this_psi_q[2]
                        psi_q_inv =  this_psi_q[3]
                        psi_q_out =  this_psi_q[4]
            
                psi_wza_con_change_par_x[count,0] = psi_wza_con
                psi_wza_kap_change_par_x[count,0] = psi_wza_kap
                psi_wza_lab_change_par_x[count,0] = psi_wza_lab
                psi_wza_inv_change_par_x[count,0] = psi_wza_inv
                psi_wza_out_change_par_x[count,0] = psi_wza_out
            
                if info=="RobustFullInfo" or info=="RobustPartialInfo":
                    psi_q_con_change_par_x[count,0] = psi_q_con
                    psi_q_kap_change_par_x[count,0] = psi_q_kap
                    psi_q_lab_change_par_x[count,0] = psi_q_lab
                    psi_q_inv_change_par_x[count,0] = psi_q_inv
                    psi_q_out_change_par_x[count,0] = psi_q_out
                            
                count = count+1
#            print "var_xpar_levpar[0, lev_par_count, :].shape", var_xpar_levpar[0, lev_par_count, :].shape    
#            print "psi_wza_con_change_par_x", psi_wza_con_change_par_x.shape            
            var_xpar_levpar[0, lev_par_count, :] = psi_wza_con_change_par_x[:,0]
            var_xpar_levpar[1, lev_par_count, :] = psi_wza_kap_change_par_x[:,0]
            var_xpar_levpar[2, lev_par_count, :] = psi_wza_lab_change_par_x[:,0]
            var_xpar_levpar[3, lev_par_count, :] = psi_wza_inv_change_par_x[:,0]
            var_xpar_levpar[4, lev_par_count, :] = psi_wza_out_change_par_x[:,0]
    
            current_label = ''.join([par_tex_lev, ' = {0:.2f}']).format(this_par_lev)
   
            ax_con_parx_parlev.plot(set_of_par_x,psi_wza_con_change_par_x, label=current_label)
            ax_kap_parx_parlev.plot(set_of_par_x,psi_wza_kap_change_par_x, label=current_label)
            ax_lab_parx_parlev.plot(set_of_par_x,psi_wza_lab_change_par_x, label=current_label)
            ax_inv_parx_parlev.plot(set_of_par_x,psi_wza_inv_change_par_x, label=current_label)
            ax_out_parx_parlev.plot(set_of_par_x,psi_wza_out_change_par_x, label=current_label)
            
            if info=="RobustFullInfo" or info=="RobustPartialInfo":
                 ax_q_con_parx_parlev.plot(set_of_par_x,psi_q_con_change_par_x, label=current_label)
                 ax_q_kap_parx_parlev.plot(set_of_par_x,psi_q_kap_change_par_x, label=current_label)
                 ax_q_lab_parx_parlev.plot(set_of_par_x,psi_q_lab_change_par_x, label=current_label)
                 ax_q_inv_parx_parlev.plot(set_of_par_x,psi_q_inv_change_par_x, label=current_label)
                 ax_q_out_parx_parlev.plot(set_of_par_x,psi_q_out_change_par_x, label=current_label)
                 
                 var_xpar_levpar_q[0, lev_par_count, :] = psi_q_con_change_par_x[:,0]
                 var_xpar_levpar_q[1, lev_par_count, :] = psi_q_kap_change_par_x[:,0]
                 var_xpar_levpar_q[2, lev_par_count, :] = psi_q_lab_change_par_x[:,0]
                 var_xpar_levpar_q[3, lev_par_count, :] = psi_q_inv_change_par_x[:,0]
                 var_xpar_levpar_q[4, lev_par_count, :] = psi_q_out_change_par_x[:,0]

            
            lev_par_count = lev_par_count+1    
                    
        
        ax_con_parx_parlev.legend(loc="best")    
        ax_kap_parx_parlev.legend(loc="best")    
        ax_lab_parx_parlev.legend(loc="best")    
        ax_inv_parx_parlev.legend(loc="best")    
        ax_out_parx_parlev.legend(loc="best")    
        
        ax_con_parx_parlev.set_xlabel(par_tex_x)    
        ax_kap_parx_parlev.set_xlabel(par_tex_x)    
        ax_lab_parx_parlev.set_xlabel(par_tex_x)    
        ax_inv_parx_parlev.set_xlabel(par_tex_x)    
        ax_out_parx_parlev.set_xlabel(par_tex_x)    
        
        ax_con_parx_parlev.set_ylabel(psiw_name)    
        ax_kap_parx_parlev.set_ylabel(psiw_name)    
        ax_lab_parx_parlev.set_ylabel(psiw_name)    
        ax_inv_parx_parlev.set_ylabel(psiw_name)    
        ax_out_parx_parlev.set_ylabel(psiw_name)    
        
        ax_con_parx_parlev.grid(True)    
        ax_kap_parx_parlev.grid(True)   
        ax_lab_parx_parlev.grid(True)   
        ax_inv_parx_parlev.grid(True)   
        ax_out_parx_parlev.grid(True)   
        
        ax_con_parx_parlev.set_title(r"Consumption: value of " + psiw_name)
        ax_kap_parx_parlev.set_title(r"Capital: value of " + psiw_name)
        ax_lab_parx_parlev.set_title(r"Labor: value of " + psiw_name)
        ax_inv_parx_parlev.set_title(r"Investment: value of " + psiw_name)
        ax_out_parx_parlev.set_title(r"Output: value of " + psiw_name)
        
        filename_con_wza =  "../../figures/psi_wza_con_change_" + par_name_x + "_" + par_name_lev + "_" + w_name + "_" + postname + ".pdf"
        filename_kap_wza =  "../../figures/psi_wza_kap_change_" + par_name_x + "_" + par_name_lev + "_" + w_name + "_" + postname +  ".pdf"
        filename_lab_wza =  "../../figures/psi_wza_lab_change_" + par_name_x + "_" + par_name_lev + "_" + w_name + "_" + postname +  ".pdf"
        filename_inv_wza =  "../../figures/psi_wza_inv_change_" + par_name_x + "_" + par_name_lev + "_" + w_name + "_" + postname +  ".pdf"
        filename_out_wza =  "../../figures/psi_wza_out_change_" + par_name_x + "_" + par_name_lev + "_" + w_name + "_" + postname +  ".pdf"
        
        fig_con_parx_parlev.savefig(filename_con_wza)
        fig_kap_parx_parlev.savefig(filename_kap_wza)
        fig_lab_parx_parlev.savefig(filename_lab_wza)
        fig_inv_parx_parlev.savefig(filename_inv_wza)
        fig_out_parx_parlev.savefig(filename_out_wza)
        
        if info=="RobustFullInfo" or info=="RobustPartialInfo":
            ax_q_con_parx_parlev.legend(loc="best")    
            ax_q_kap_parx_parlev.legend(loc="best")    
            ax_q_lab_parx_parlev.legend(loc="best")    
            ax_q_inv_parx_parlev.legend(loc="best")    
            ax_q_out_parx_parlev.legend(loc="best")    
            
            ax_q_con_parx_parlev.set_title(r"Consumption: value of $\psi_{q}$")
            ax_q_kap_parx_parlev.set_title(r"Capital: value of $\psi_{q}$")
            ax_q_lab_parx_parlev.set_title(r"Labor: value of $\psi_{q}$")
            ax_q_inv_parx_parlev.set_title(r"Investment: value of $\psi_{q}$")
            ax_q_out_parx_parlev.set_title(r"Output: value of $\psi_{q}$")
            
            ax_q_con_parx_parlev.set_xlabel(par_tex_x)    
            ax_q_kap_parx_parlev.set_xlabel(par_tex_x)    
            ax_q_lab_parx_parlev.set_xlabel(par_tex_x)    
            ax_q_inv_parx_parlev.set_xlabel(par_tex_x)    
            ax_q_out_parx_parlev.set_xlabel(par_tex_x)    
        
            ax_q_con_parx_parlev.set_ylabel(psiw_name)    
            ax_q_kap_parx_parlev.set_ylabel(psiw_name)    
            ax_q_lab_parx_parlev.set_ylabel(psiw_name)    
            ax_q_inv_parx_parlev.set_ylabel(psiw_name)    
            ax_q_out_parx_parlev.set_ylabel(psiw_name)    

            filename_con_q =  "../../figures/psi_q_con_change_" + par_name_x + "_" + par_name_lev + "_" + w_name + "_" + postname +  ".pdf"
            filename_kap_q =  "../../figures/psi_q_kap_change_" + par_name_x + "_" + par_name_lev + "_" + w_name + "_" + postname +  ".pdf"
            filename_lab_q =  "../../figures/psi_q_lab_change_" + par_name_x + "_" + par_name_lev + "_" + w_name + "_" + postname +  ".pdf"
            filename_inv_q =  "../../figures/psi_q_inv_change_" + par_name_x + "_" + par_name_lev + "_" + w_name + "_" + postname +  ".pdf"
            filename_out_q =  "../../figures/psi_q_out_change_" + par_name_x + "_" + par_name_lev + "_" + w_name + "_" + postname +  ".pdf"
            
            fig_q_con_parx_parlev.savefig(filename_con_q)
            fig_q_kap_parx_parlev.savefig(filename_kap_q)
            fig_q_lab_parx_parlev.savefig(filename_lab_q)
            fig_q_inv_parx_parlev.savefig(filename_inv_q)
            fig_q_out_parx_parlev.savefig(filename_out_q)
            
            
                    
        info_count=info_count+1
    
    if "RobustFullInfo" in set_of_info_types  or "RobustPartialInfo" in set_of_info_types :
        return var_xpar_levpar, var_xpar_levpar_q
    else:
        return var_xpar_levpar
        
        


def all_vars_par_par(a_of_values, set_par_x, set_par_lev, par_tex_x, 
                     par_tex_lev, par_name_x, par_name_lev, vars_tex, postname):

    for v in range(len(set_par_lev)):
        fig, ax = plt.subplots() 
        for i in range(5):
            ax.plot(set_par_x, a_of_values[i,v,:], label=vars_tex[i])
            
        ax.legend(loc="best", ncol=2)    
        ax.grid(True)  
        ax.set_title("Coeff of persistent shock " +
                            "("+par_tex_lev + " = " + 
                            "{0:.2f})".format(set_par_lev[v]) )
        ax.set_xlabel(par_tex_x)
        filename =  "../../figures/psi_change_" + par_name_x + "_" + par_name_lev + "_all_vars_" + str(v) + "_" + postname + ".pdf"
        fig.savefig(filename)

def ir_toy_rpi_from(azA_mat, x_sta, x_nsta, BGP_det, phi_q, islog=False):
    
    copy_phi_q =  phi_q.copy()
    copy_phi_q.shape = (len(phi_q), 1)
    
    x_sta_plus_phiq =  x_sta + copy_phi_q
    
    return x_sta_plus_phiq, azA_mat, x_sta, x_nsta, BGP_det
    