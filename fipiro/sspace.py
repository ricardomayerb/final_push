import utils as u

class SignalStateSpace:

    def __init__(self, n_x=1, n_s=1, param_val_dic={},
                 user_names_dic={}):

        ssp_mats_sym_d = u.make_state_space_sym(n_s, n_x, True)

        self.A_sym = ssp_mats_sym_d['A_z']
        self.C_sym = ssp_mats_sym_d['C_z']
        self.D_sym = ssp_mats_sym_d['D_s']
        self.G_sym = ssp_mats_sym_d['G_s']

        if user_names_dic != {}:
            self.A_sym_u = self.A_sym.subs(user_names_dic)
            self.C_sym_u = self.C_sym.subs(user_names_dic)
            self.D_sym_u = self.D_sym.subs(user_names_dic)
            self.G_sym_u = self.G_sym.subs(user_names_dic)
            self.A_num = self.A_sym_u.subs(param_val_dic)
            self.A_num = u.matrix2numpyfloat(self.A_num)
            self.C_num = self.C_sym_u.subs(param_val_dic)
            self.C_num = u.matrix2numpyfloat(self.C_num)
            self.D_num = self.D_sym_u.subs(param_val_dic)
            self.D_num = u.matrix2numpyfloat(self.D_num)
            self.G_num = self.G_sym_u.subs(param_val_dic)
            self.G_num = u.matrix2numpyfloat(self.G_num)

        else:
            self.A_num = self.A_sym.subs(param_val_dic)
            self.A_num = u.matrix2numpyfloat(self.A_num)
            self.C_num = self.C_sym.subs(param_val_dic)
            self.C_num = u.matrix2numpyfloat(self.C_num)
            self.D_num = self.D_sym.subs(param_val_dic)
            self.D_num = u.matrix2numpyfloat(self.D_num)
            self.G_num = self.G_sym.subs(param_val_dic)
            self.G_num = u.matrix2numpyfloat(self.G_num)
            self.A_sym_u = {}
            self.C_sym_u = {}
            self.D_sym_u = {}
            self.G_sym_u = {}
