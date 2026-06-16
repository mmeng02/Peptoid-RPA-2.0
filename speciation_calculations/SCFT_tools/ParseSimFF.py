import numpy as np
import re

class SimForceField:

    def __init__(self, ff_file, beadtypes): 

        self.beadtypes = beadtypes
        self.ff_file = ff_file
        self.kappa_matrix = np.zeros((len(beadtypes),len(beadtypes)))
        self.B_matrix = np.zeros((len(beadtypes),len(beadtypes)))
        self.excluded_volume_matrix = np.zeros((len(beadtypes),len(beadtypes)))
        self.Dist0_matrix = np.zeros((len(beadtypes),len(beadtypes)))
        self.FConst_matrix = np.zeros((len(beadtypes),len(beadtypes)))
        self.lb_matrix = np.zeros((len(beadtypes),len(beadtypes)))
        self.ff_dictionary = {}


    def CreateFFDictionary(self):

        fr = open(self.ff_file,'r')
        frlines = fr.readlines()

        for i,line in enumerate(frlines):
            l = line.split()
            if l == []:
                continue

            if l[1] == 'POTENTIAL': 
                txt = l[2].split('_')
                if len(txt) == 1:
                    continue

                parameters = {}
                l1 = frlines[i+1]
                l1 = re.sub('[^0-9,A-Z,a-z,.,-]', ' ', l1)
                l1 = re.sub(',', ' ', l1).split()
                for j in range(0,len(l1),2):
                    parameters[l1[j]] = l1[j+1]

                name = '_'.join(txt[0:-2])
                
                self.ff_dictionary['_'.join([name,txt[-2],txt[-1]])] = parameters
                #self.ff_dictionary['_'.join([name,txt[-1],txt[-2]])] = parameters

        print('\n========== CG Force Field Definitions ==========\n')     
        for key,values in self.ff_dictionary.items():
            print(key, ' : ', values)  
        return
    

    def CreateKappaMatrix(self):

        for i in range(len(self.beadtypes)):
            for j in range(len(self.beadtypes)):
                bi = self.beadtypes[i][0]
                bj = self.beadtypes[j][0]
                try: 
                    kappa = float(self.ff_dictionary['ljg_{}_{}'.format(bi,bj)]['Kappa'])
                except:
                    kappa = float(self.ff_dictionary['ljg_{}_{}'.format(bj,bi)]['Kappa'])
                self.kappa_matrix[i][j] = kappa

        print('\n========== Creating Kappa (1 / (2 * ai^2 + 2 * aj^2) Matrix ==========\n') 

        print('bead species : ', [b[0] for b in self.beadtypes])

        print('\n', self.kappa_matrix)
        return
    

    def CreateExcludedVolumeMatrix(self):

        self.CreateKappaMatrix()

        for i in range(len(self.beadtypes)):
            for j in range(len(self.beadtypes)):
                bi = self.beadtypes[i][0]
                bj = self.beadtypes[j][0]
                kappa = self.kappa_matrix[i][j]
                try: 
                    B_excl_vol = float(self.ff_dictionary['ljg_{}_{}'.format(bi,bj)]['B'])
                except:
                    B_excl_vol = float(self.ff_dictionary['ljg_{}_{}'.format(bj,bi)]['B'])
                u_excl_vol = B_excl_vol * (np.pi * kappa**(-1)) ** (3 / 2)
                self.B_matrix[i][j] = B_excl_vol
                self.excluded_volume_matrix[i][j] = u_excl_vol

        print('\n========== Creating Excluded Volume Matrix ==========') 

        print('\n bead species : ', [b[0] for b in self.beadtypes])

        print('\n', self.excluded_volume_matrix)
        return
    

    def CreateBondMatrix(self):

        for i in range(len(self.beadtypes)):
            for j in range(len(self.beadtypes)):
                bi = self.beadtypes[i][0]
                bj = self.beadtypes[j][0]

                try:
                    Dist0 = float(self.ff_dictionary['bond_{}_{}'.format(bi,bj)]['Dist0'])
                    FConst = float(self.ff_dictionary['bond_{}_{}'.format(bi,bj)]['FConst'])
                except:
                    try:
                        Dist0 = float(self.ff_dictionary['bond_{}_{}'.format(bj,bj)]['Dist0'])
                        FConst = float(self.ff_dictionary['bond_{}_{}'.format(bj,bi)]['FConst'])
                    except:
                        FConst = 0.0
                        Dist0 = 0.0
                
                self.Dist0_matrix[i][j] = Dist0
                self.FConst_matrix[i][j] = FConst

        print('\n========== Creating Bond Matrices ==========\n') 

        print('bead species : ', [b[0] for b in self.beadtypes])

        print('\n', self.Dist0_matrix)
        print('\n', self.FConst_matrix)

        return


    def CreateEwaldMatrix(self):

        for i in range(len(self.beadtypes)):
            for j in range(len(self.beadtypes)):
                bi = self.beadtypes[i][0]
                bj = self.beadtypes[j][0]

                if 'smeared_corr_{}_{}'.format(bi,bj)  in self.ff_dictionary.keys():
                    lb = float(self.ff_dictionary['smeared_corr_{}_{}'.format(bi,bj)]['Coef'])
                else:
                    lb = 0.0
                
                self.lb_matrix[i][j] = lb

        print('\n========== Creating Ewald Matrix ==========\n') 

        print('bead species : ', [b[0] for b in self.beadtypes])

        print('\n', self.lb_matrix)

        return

    def WriteForceFieldField(self, filename=None):

        fi = open(filename,'w')
        s = ''
        for key, values in self.ff_dictionary.items():
            s += '>>> POTENTIAL {}\n'.format(key)
            s += '{'
            for i, (_k, _v) in enumerate(values.items()):
                s+= '\'{}\': {}'.format(_k, _v)
                if i < len(values)-1:
                    s +=', '
            s += '}\n'
        
        fi.write(s)
        fi.close()

        return
        

