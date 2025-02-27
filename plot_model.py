import numpy as np
import matplotlib.pyplot as plt
import os
from COG import Voigt
from spectrum_model import spectrum
from spectro.a_unc import a
from collections import OrderedDict
from spectro.profiles import convolve_res2
from scipy.interpolate import interp1d
import pickle
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import matplotlib.patches as mpatches

if 1:
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    #matplotlib.rcParams['text.latex.unicode'] = True
    #matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['axes.titlesize'] = 10
    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'

class par:
    def __init__(self, parent, name, val, min, max, step, addinfo='', vary=True, fit=True, show=True, left=None, right=None):
        self.parent = parent
        self.name = name
        if 'cont' in self.name:
            self.dec = 4
        elif 'me' in self.name:
            self.dec = 3
        elif 'res' in self.name:
            self.dec = 1
        elif 'cf' in self.name:
            self.dec = 3
        elif 'dispz' in self.name:
            self.dec = 1
        elif 'disps' in self.name:
            self.dec = 8
        elif 'sts' in self.name:
            self.dec = 3
        elif 'stN' in self.name:
            self.dec = 2
        else:
            d = {'z': 7, 'b': 3, 'N': 3, 'turb': 3, 'kin': 2, 'mu': 8, 'iso': 3, 'hcont': 3,
                 'Ntot': 3, 'logn': 3, 'logT': 3, 'logf': 3, 'rad': 3, 'CMB': 3}
            self.dec = d[self.name]

        if self.name in ['N', 'Ntot', 'logn', 'logT', 'rad', 'iso', 'me', 'mu', 'sts', 'stNl', 'stNu']:
            self.form = 'l'
        else:
            self.form = 'd'

        if self.name in ['b', 'N']:
            self.sys = self.parent.parent
        elif self.name in ['z', 'turb', 'kin', 'Ntot', 'logn', 'logT', 'logf', 'rad', 'CMB']:
            self.sys = self.parent
        else:
            self.sys = None

        self.val = val
        self.min = min
        self.max = max
        self.step = step
        self.vary = vary
        self.addinfo = addinfo
        self.fit = fit
        self.fit_w = self.fit
        self.show = show
        self.left = left
        self.right = right
        self.unc = a()

    def set(self, val, attr='val', check=True):
        if attr == 'unc':
            if isinstance(val, (int, float)):
                setattr(self, attr, a(self.val, val, self.form))
            elif isinstance(val, str):
                setattr(self, attr, a(val, self.form))
            else:
                setattr(self, attr, val)
        else:
            setattr(self, attr, val)
            if attr == 'vary':
                self.fit = self.vary
        if attr == 'val' and check:
            return self.check_range()

    def check_range(self):

        if self.val < self.min:
            self.val = self.min
            return False

        if self.val > self.max:
            self.val = self.max
            return False

        return True

    def check(self):
        if self.addinfo == '':
            self.vary = True
        else:
            self.vary = False

    def duplicate(self, other):
        attrs = ['val', 'min', 'max', 'step', 'vary', 'addinfo', 'fit', 'fit_w']
        for attr in attrs:
            setattr(self, attr, getattr(other, attr))

    def copy(self):
        return par(self.parent, self.name, self.val, self.min, self.max, self.step, addinfo=self.addinfo, vary=self.vary, fit=self.fit, show=self.show)

    def latexname(self):
        pass

    def ref(self, val=None, attr='val'):
        # special function for lmfit to translate redshift to velocity space
        if self.name.startswith('z'):
            c = 299792.458
            if 1:
                if val is None:
                    self.saved = self.val
                    return c * (self.val - self.saved), c * (self.min - self.saved), c * (self.max - self.saved)
                else:
                    if attr in ['val', 'min', 'max']:
                        return self.saved + val / c
                    elif attr in ['step', 'unc']:
                        return val / c
            else:
                if val is None:
                    self.saved = self.val
                    return self.val / self.saved, self.min / self.saved, self.max / self.saved
                else:
                    if attr in ['val', 'min', 'max']:
                        return val * self.saved
                    elif attr in ['step', 'unc']:
                        return val * self.saved
        else:
            if val is None:
                return self.val, self.min, self.max
            else:
                return val

    def __repr__(self):
        s = self.name
        if self.name in ['z', 'b', 'N', 'turb', 'kin', 'Ntot', 'logn', 'logT', 'logf', 'rad', 'CMB']:
            s += '_' + str(self.sys.ind)
        if self.name in ['b', 'N']:
            s += '_' + self.parent.name
        return s

    def __str__(self):
        s = self.name
        if self.name in ['z', 'b', 'N', 'turb', 'kin', 'Ntot', 'logn', 'logT', 'logf', 'rad', 'CMB']:
            s += '_' + str(self.sys.ind)
        if self.name in ['b', 'N']:
            s += '_' + self.parent.name
        return s

    def str(self, attr=None):
        if attr is None:
            if 'cf' not in self.name:
                return '{1:} {2:.{0}f} {3:.{0}f} {4:.{0}f} {5:.{0}f} {6:1d} {7:s}'.format(self.dec, self, self.val, self.min, self.max, self.step, self.vary, self.addinfo)
            else:
                return '{1:} {2:.{0}f} {3:.{0}f} {4:.{0}f} {5:.{0}f} {6:1d} {7:s}'.format(self.dec, self, self.val, self.left, self.right, self.step, self.vary, self.addinfo)
        if attr == 'lmfit':
            return '{1:} {2:.{0}f} ± {3:.{0}f}'.format(self.dec, self, self.val, self.step)
        else:
            return '{0:.{1}f}'.format(getattr(self, attr), self.dec)

    def fitres(self, latex=False, dec=None, showname=True, classview=False, aview=False):
        if self.unc is not None:
            if dec is None and not latex:
                d = np.asarray([self.unc.plus, self.unc.minus])
                if len(np.nonzero(d)[0]) > 0:
                    dec = int(np.round(np.abs(np.log10(np.min(d[np.nonzero(d)])))) + 1)
                else:
                    dec = self.dec
            if latex:
                if self.name in ['z']:
                    dec = int(np.round(np.abs(np.log10(np.min(np.asarray([self.unc.plus, self.unc.minus])[np.nonzero(np.asarray([self.unc.plus, self.unc.minus]))])))) + 1)
                    return '${0:.{3}f}(^{{+{1:d}}}_{{-{2:d}}})$'.format(self.unc.val, int(self.unc.plus*10**dec), int(self.unc.minus*10**dec), dec)
                else:
                    return self.unc.latex(f=dec, base=0)
                    #return '${0:.{3}f}^{{+{1:.{3}f}}}_{{-{2:.{3}f}}}$'.format(self.unc.val, self.unc.plus, self.unc.minus, dec)
            elif classview:
                if self.name in ['z']:
                    return 'co = sy({0:.{2}f}, {1:d})'.format(self.unc.val, int(np.sqrt(self.unc.plus**2 + self.unc.minus**2) * 10 ** dec), dec)
                elif self.name in ['N']:
                    return "co.el('{0}', {1:.{4}f}, {2:.{4}f}, {3:.{4}f})".format(self.parent.name, self.unc.val, self.unc.plus, self.unc.minus, dec)
            elif aview:
                return '({0:.{3}f}, {1:.{3}f}, {2:.{3}f})'.format(self.unc.val, self.unc.plus, self.unc.minus, dec)
            else:
                return '{0} = {1:.{4}f} + {2:.{4}f} - {3:.{4}f}'.format(str(self), self.unc.val, self.unc.plus, self.unc.minus, dec)
        else:
            if dec is None:
                dec = self.dec
            if showname:
                return '{0} = {1:.{2}f}'.format(str(self), self.val, dec)
            else:
                return '{0:.{1}f}'.format(self.val, dec)

class fitSpecies:
    def __init__(self, parent, name=None):
        self.parent = parent
        self.name = name
        #self.mass = self.setmass(name)
        self.b = par(self, 'b', 4, 0.5, 200, 0.05)
        self.N = par(self, 'N', 14, 10, 22, 0.01)

    def duplicate(self, other):
        attrs = ['b', 'N']
        for attr in attrs:
            getattr(self, attr).duplicate(getattr(other, attr))

class fitSystem:
    def __init__(self, parent, z=0.0):
        self.parent = parent
        self.z = par(self, 'z', z, z-0.001, z+0.001, 1e-7)
        #self.cons_vary = False
        #self.turb = par(self, 'turb', 5, 0.5, 20, 0.05, vary=self.cons_vary, fit=self.cons_vary)
        #self.kin = par(self, 'kin', 5e4, 1e4, 1e5, 1e3, vary=self.cons_vary, fit=self.cons_vary)
        self.sp = OrderedDict()
        self.total = OrderedDict()
        self.pr = None
        self.exclude = []

    def add(self, name):
        if name in 'turb':
            self.turb = par(self, 'turb', 5, 0.5, 20, 0.05)
        if name in 'kin':
            self.kin = par(self, 'kin', 5e3, 1e3, 3e4, 1e3)
        if name in 'Ntot':
            self.Ntot = par(self, 'Ntot', 14, 12, 22, 0.05)
        if name in 'logn':
            self.logn = par(self, 'logn', 2, -2, 5, 0.05)
        if name in 'logT':
            self.logT = par(self, 'logT', 2, 0.5, 5, 0.05)
        if name in 'logf':
            self.logf = par(self, 'logf', 0, -6, 0, 0.05)
        if name in 'rad':
            self.rad = par(self, 'rad', 0, -6, 30, 0.05)
        if name in 'CMB':
            self.CMB = par(self, 'CMB', 2.726 * (1 + self.z.val), 0, 50, 0.05)

    def remove(self, name):
        if name in ['turb', 'kin', 'Ntot', 'logn', 'logT', 'logf', 'rad', 'CMB']:
            if hasattr(self, name):
                delattr(self, name)

    def addSpecies(self, name, dic='sp'):
        if name not in getattr(self, dic).keys():
            getattr(self, dic)[name] = fitSpecies(self, name)
            #if self.parent.parent is not None:
            #    self.parent.parent.console.exec_command('show ' + name)
            return True
        else:
            return False

    def duplicate(self, other):
        self.z.duplicate(other.z)
        attrs = ['turb', 'kin', 'Ntot', 'logn', 'logT', 'logf', 'rad', 'CMB']
        for attr in attrs:
            if hasattr(other, attr):
                self.add(attr)
                getattr(self, attr).duplicate(getattr(other, attr))
        self.sp = OrderedDict()
        for k, v in other.sp.items():
            self.sp[k] = fitSpecies(self, name=k)
            self.sp[k].duplicate(v)

    def zshift(self, v):
        self.z.val += float(v) / 299792.458 * (1 + self.z.val)

    def zrange(self, v):
        self.z.min = self.z.val - abs(float(v)) / 299792.458 * (1 + self.z.val)
        self.z.max = self.z.val + abs(float(v)) / 299792.458 * (1 + self.z.val)

    def N(self, sp):
        if sp in self.sp.keys():
            if hasattr(self.parent, 'me') and 'HI' != sp and 'HI' in self.sp.keys():
                return abundance(sp, self.sp['HI'].N.val, self.parent.me.val)
            else:
                return self.sp[sp].N.val

    def pyratio(self, init=False):
        #t = Timer('pyratio '+ str(self.parent.sys.index(self)))
        if init or self.pr is None:
            if self == self.parent.sys[0]:

                self.pr = pyratio(z=self.z.val, pumping='simple', radiation='simple', sed_type=self.rad.addinfo)

                #print('init', self.pr.pumping, self.pr.radiation,  self.pr.sed_type)
                d = {'CO': [-1, 10], 'CI': [-1, 3], 'FeII': [-1, 13], 'H2': [-1, 3]}
                for s in self.sp.keys():
                    print(s)
                    if s.startswith('CO'):
                        d['CO'][0] = 0 if s[3:4].strip() == '' else max(d['CO'][0], int(s[3:4]))
                        pars = ['T', 'n', 'f', 'CMB']
                        self.pr = pyratio(z=self.z.val, radiation='full', sed_type='CMB')
                    if s.startswith('CI'):
                        d['CI'][0] = 0 if s[3:4].strip() == '' else max(d['CI'][0], int(s[3:4]))
                        pars = ['T', 'n', 'f', 'rad', 'CMB']
                    if 'FeII' in s:
                        d['FeII'][0] = 0 if s[5:6].strip() == '' else max(d['FeII'][0], int(s[5:6]))
                        pars = ['T', 'e', 'rad']
                    if 'H2' in s:
                        d['H2'][0] = 0 if s[3:4].strip() == '' else max(d['H2'][0], int(s[3:4]))
                        pars = ['T', 'n', 'f', 'rad']
                self.pr.set_pars(pars)
                print(d)
                for k, v in d.items():
                    if v[0] > -1:
                        self.pr.add_spec(k, num=v[1])
            else:
                self.pr = deepcopy(self.parent.sys[0].pr)

        #t.time('init')
        if self.pr is not None:
            self.pr.pars['T'].value = self.logT.val
            if 'n' in self.pr.pars.keys():
                self.pr.pars['n'].value = self.logn.val
            if 'e' in self.pr.pars.keys():
                self.pr.pars['e'].value = self.logn.val
            if 'f' in self.pr.pars.keys():
                self.pr.pars['f'].value = self.logf.val
            if 'rad' in self.pr.pars.keys():
                self.pr.pars['rad'].value = self.rad.val
            if 'CMB' in self.pr.pars.keys():
                self.pr.pars['CMB'].value = self.CMB.val
            for k in self.pr.species.keys():
                col = self.pr.predict(name=k, level=-1, logN=self.Ntot.val)
                for s in self.sp.keys():
                    if k in s and 'Ntot' in self.sp[s].N.addinfo:
                        self.sp[s].N.val = col[self.pr.species[k].names.index(s)]
        #t.time('predict')


    def __str__(self):
        return '{:.6f} '.format(self.z.val) + str(self.sp)

class fitPars:
    def __init__(self, parent):
        self.parent = parent
        self.sys = []
        self.me_num = 0
        self.res_num = 0
        self.cf_fit = False
        self.cf_num = 0
        self.disp_num = 0
        self.stack_num = 0
        self.tieds = {}

    def add(self, name, addinfo=''):
        if name in 'mu':
            self.mu = par(self, 'mu', 1e-6, 1e-7, 5e-6, 1e-8)
        if 'me' in name:
            setattr(self, name, par(self, name, 0, -3, 1, 0.01))
        print(name)
        if name in 'iso':
            print('iso', name)
            if addinfo == 'D/H':
                self.iso = par(self, 'iso', -4.5, -5.4, -4, 0.01, addinfo=addinfo)
            if addinfo == '13C/12C':
                self.iso = par(self, 'iso', -2.0, -5, 0, 0.1, addinfo=addinfo)
        if 'res' in name:
            setattr(self, name, par(self, name, 45000, 1000, 60000, 1, addinfo='exp_0'))
        if 'cont' in name:
            self.cont.add(name)
        if name in 'hcont':
            self.hcont = par(self, 'hcont', 1, 0, 10, 0.1)
        if 'cf' in name:
            setattr(self, name, par(self, name, 0.1, 0, 1, 0.01, addinfo='all', left=3000, right=9000))
        if 'dispz' in name:
            setattr(self, name, par(self, name, 5000, 3000, 9000, 0.1, addinfo='exp_0'))
        if 'disps' in name:
            setattr(self, name, par(self, name, 1e-5, -1e-4, 1e-4, 1e-6, addinfo='exp_0'))
        if 'sts' in name:
            setattr(self, name, par(self, name, -1.5, -2, -1, 0.01))
        if 'stNl' in name:
            setattr(self, name, par(self, name, 18, 16, 20, 0.05))
        if 'stNu' in name:
            setattr(self, name, par(self, name, 21, 20, 22, 0.05))


    def remove(self, name):
        if name in ['mu', 'iso', 'hcont'] or any([x in name for x in ['me', 'res', 'cf', 'disp', 'sts', 'stNu', 'stNl']]):
            if hasattr(self, name):
                delattr(self, name)
        if 'cont_' in name:
            self.cont.remove(name)
                #gc.collect()

    def addTieds(self, p1, p2):
        try:
            self.getPar(p1)
            self.getPar(p2)
            if not (p1 in self.tieds.keys() and self.tieds[p1] == p2):
                self.tieds[p1] = p2
                self.setValue(p1, False, 'vary')
        except:
            pass

    def addSys(self, ind=-1, z=None):
        if z is None:
            if len(self.sys) > 0:
                self.sys.append(fitSystem(self, 0))
                self.sys[-1].duplicate(self.sys[ind])
            else:
                self.sys.append(fitSystem(self, 0))
        else:
            self.sys.append(fitSystem(self, z))
        self.refreshSys()

    def delSys(self, ind=-1):

        if 1:
            if ind < len(self.sys):
                for i in range(ind, len(self.sys)-1):
                    self.swapSys(i, i+1)

            ind = len(self.sys)-1
        s = self.sys[ind]
        self.sys.remove(s)
        del s

        gc.collect()
        self.refreshSys()

    def swapSys(self, i1, i2):
        print(i1, i2)
        if self.cf_fit:
            for i in range(self.cf_num):
                if hasattr(self, 'cf_' + str(i)):
                    p = getattr(self, 'cf_' + str(i))
                    cf = p.addinfo.split('_')
                    print('cf', cf)
                    if cf[0].find('sys') > -1:
                        if i1 in [int(s) for s in cf[0].split('sys')[1:]]:
                            self.setValue('cf_' + str(i), 'sys'+'sys'.join([s if int(s) != i1 else str(i2) for s in cf[0].split('sys')[1:]]) + '_' + cf[1], 'addinfo')
                        if i2 in [int(s) for s in cf[0].split('sys')[1:]]:
                            self.setValue('cf_' + str(i), 'sys'+'sys'.join([s if int(s) != i2 else str(i1) for s in cf[0].split('sys')[1:]]) + '_' + cf[1], 'addinfo')
                        #if int(p.addinfo[p.addinfo.find('sys')+3:p.addinfo.find('_')]) == i1:
                        #    p.addinfo = p.addinfo[:p.addinfo.find('sys')+3]+str(i2)+p.addinfo[p.addinfo.find('_'):]
                        #elif int(p.addinfo[p.addinfo.find('sys')+3:p.addinfo.find('_')]) == i2:
                        #    p.addinfo = p.addinfo[:p.addinfo.find('sys')+3]+str(i1)+p.addinfo[p.addinfo.find('_'):]
        self.sys[i1], self.sys[i2] = self.sys[i2], self.sys[i1]
        self.refreshSys()

    def refreshSys(self):
        for i, s, in enumerate(self.sys):
            s.ind = i

    def setValue(self, name, val, attr='val', check=True):
        s = name.split('_')
        if attr in ['val', 'min', 'max', 'step', 'left', 'right']:
            val = float(val)
        elif attr in ['vary', 'fit']:
            val = int(val)

        if s[0] in ['mu', 'iso']:
            if not hasattr(self, s[0]):
                self.add(s[0])
            res = getattr(self, s[0]).set(val, attr, check=check)

        if s[0] in ['dtoh']:
            if not hasattr(self, 'iso'):
                self.add('iso', addinfo='D/H')
            res = getattr(self, 'iso').set(val, attr, check=check)

        if s[0] in ['me', 'res', 'cf', 'dispz', 'disps', 'hcont', 'sts', 'stNl', 'stNu']:
            if not hasattr(self, name):
                self.add(name)
            res = getattr(self, name).set(val, attr, check=check)

        if s[0] in ['cont']:
            if not hasattr(self, name):
                self.cont.add(name)
            res = getattr(self, name).set(val, attr, check=check)

        if s[0] in ['z', 'turb', 'kin', 'Ntot', 'logn', 'logT', 'logf', 'rad', 'CMB']:
            while len(self.sys) <= int(s[1]):
                self.addSys()
            if s[0] in ['turb', 'kin', 'Ntot', 'logn', 'logT', 'logf', 'rad', 'CMB']:
                if not hasattr(self.sys[int(s[1])], s[0]):
                    self.sys[int(s[1])].add(s[0])
            res = getattr(self.sys[int(s[1])], s[0]).set(val, attr, check=check)

        if s[0] in ['b', 'N']:
            while len(self.sys) <= int(s[1]):
                self.addSys()
            self.sys[int(s[1])].addSpecies(s[2])
            res = getattr(self.sys[int(s[1])].sp[s[2]], s[0]).set(val, attr, check=check)

        return res

    def update(self, what='all', ind='all', redraw=True):

        if what in ['all', 'res']:
            if self.res_num > 0:
                for i in range(self.res_num):
                    if i < len(self.parent.s):
                        self.parent.s[int(getattr(self, 'res_'+str(i)).addinfo[4:])].resolution = self.getValue('res_'+str(i))

        if what in ['all', 'cf']:
            if redraw and self.cf_fit:
                for i in range(self.cf_num):
                    try:
                        self.parent.plot.pcRegions[i].updateFromFit()
                    except:
                        pass

        for i, sys in enumerate(self.sys):
            if ind == 'all' or i == ind:
                for k, s in sys.sp.items():
                    if what in ['all', 'b', 'turb', 'kin']:
                        if s.b.addinfo != '' and s.b.addinfo != 'consist':
                            s.b.val = sys.sp[s.b.addinfo].b.val
                        elif s.b.addinfo == 'consist':
                            s.b.val = doppler(k, sys.turb.val, sys.kin.val)

                if what in ['all', 'Ntot', 'logn', 'logT', 'logf', 'rad', 'CMB']:
                    if hasattr(sys, 'Ntot'):
                        sys.pyratio()
                        what = 'all'

                for k, s in sys.sp.items():
                    if what in ['all', 'me', 'iso']:
                        if 'me' in s.N.addinfo and 'HI' in sys.sp.keys():
                            if any([sp not in k for sp in ['DI', '13C']]) and self.me_num > 0:
                                s.N.val = abundance(k, sys.sp['HI'].N.val, getattr(self, s.N.addinfo).val)
                        if s.N.addinfo == 'iso':
                            if self.iso.addinfo == 'D/H' and 'HI' in sys.sp.keys():
                                if 'DI' in k:
                                    s.N.val = sys.sp['HI'].N.val + self.iso.val
                            if self.iso.addinfo == '13C/12C':
                                if '13CI' in k and k.replace('13', '') in sys.sp.keys():
                                    s.N.val = sys.sp[k.replace('13', '')].N.val + self.iso.val


        for k, v in self.tieds.items():
            self.setValue(k, self.getValue(v))

    def getPar(self, name):
        s = name.split('_')
        par = None
        if s[0] in ['mu', 'iso', 'hcont']:
            if hasattr(self, s[0]):
                par = getattr(self, s[0])

        if s[0] in ['me', 'cont', 'res', 'cf', 'dispz', 'disps', 'sts', 'stNl', 'stNu']:
            if hasattr(self, name):
                par = getattr(self, name)

        if s[0] in ['z', 'turb', 'kin', 'Ntot', 'logn', 'logT', 'logf', 'rad', 'CMB']:
            if len(self.sys) > int(s[1]) and hasattr(self.sys[int(s[1])], s[0]):
                par = getattr(self.sys[int(s[1])], s[0])

        if s[0] in ['b', 'N']:
            if len(self.sys) > int(s[1]) and s[2] in self.sys[int(s[1])].sp and hasattr(self.sys[int(s[1])].sp[s[2]], s[0]):
                par = getattr(self.sys[int(s[1])].sp[s[2]], s[0])

        if par is None:
            raise ValueError('Fit model has no {:} parameter'.format(name))
        else:
            return par

    def getValue(self, name, attr='val'):
        par = self.getPar(name)
        if par is None:
            raise ValueError('Fit model has no {:} parameter'.format(name))
        else:
            return getattr(par, attr)

    def list(self, ind=None):
        return list(self.pars(ind).values())

    def list_check(self):
        for par in self.list():
            par.check()

    def list_fit(self):
        return [par for par in self.list() if par.fit & par.vary]

    def list_vary(self):
        return [par for par in self.list() if par.vary]

    def list_names(self):
        return [str(par) for par in self.list()]

    def list_species(self):
        species = set()
        for sys in self.sys:
            species.update(list(sys.sp.keys()))

        return species

    def pars(self, ind=None):
        pars = OrderedDict()
        if ind in [None, -1]:
            for attr in ['mu', 'iso', 'hcont']:
                if hasattr(self, attr):
                    p = getattr(self, attr)
                    pars[str(p)] = p
            if self.cont_fit and self.cont_num > 0:
                for i in range(self.cont_num):
                    for k in range(self.cont[i].num):
                        attr = 'cont_' + str(i) + '_' + str(k)
                        if hasattr(self, attr):
                            p = getattr(self, attr)
                            pars[str(p)] = p
            if self.me_num > 0:
                for i in range(self.me_num):
                    attr = 'me_' + str(i)
                    if hasattr(self, attr):
                        p = getattr(self, attr)
                        pars[str(p)] = p
            if self.res_num > 0:
                for i in range(self.res_num):
                    attr = 'res_' + str(i)
                    if hasattr(self, attr):
                        p = getattr(self, attr)
                        pars[str(p)] = p
            if self.cf_fit and self.cf_num > 0:
                for i in range(self.cf_num):
                    attr = 'cf_' + str(i)
                    if hasattr(self, attr):
                        p = getattr(self, attr)
                        pars[str(p)] = p
            if self.disp_num > 0:
                for i in range(self.disp_num):
                    for attr in ['dispz', 'disps']:
                        attr = attr + '_' + str(i)
                        if hasattr(self, attr):
                            p = getattr(self, attr)
                            pars[str(p)] = p
            if self.stack_num > 0:
                for i in range(self.stack_num):
                    for attr in ['sts', 'stNl', 'stNu']:
                        attr = attr + '_' + str(i)
                        if hasattr(self, attr):
                            p = getattr(self, attr)
                            pars[str(p)] = p
        if len(self.sys) > 0:
            for i, sys in enumerate(self.sys):
                if ind in [None, i]:
                    for attr in ['z', 'turb', 'kin', 'Ntot', 'logn', 'logT', 'logf', 'rad', 'CMB']:
                        if hasattr(sys, attr):
                            p = getattr(sys, attr)
                            pars[str(p)] = p
                    for sp in sys.sp.values():
                        for attr in ['b', 'N']:
                            if hasattr(sp, attr):
                                p = getattr(sp, attr)
                                pars[str(p)] = p
        return pars


    def list_total(self):
        pars = OrderedDict()
        for sys in self.sys:
            for k, v in sys.total.items():
                pars['_'.join(['N', str(self.sys.index(sys)), k])] = v.N
        for k, v in self.total.sp.items():
            pars['_'.join(['N', 'total', k])] = v.N

        return pars

    def readPars(self, name):
        if name.count('*') > 0:
            name = name.replace('*' * name.count('*'), 'j' + str(name.count('*')))
        s = name.split()
        attrs = ['val', 'min', 'max', 'step', 'vary', 'addinfo']
        if len(s) == len(attrs):
            s.append('')

        if 'cont_' in s[0]:
            print(self.cont_num, int(s[0].split('_')[1]))
            self.cont_num = max(self.cont_num, int(s[0].split('_')[1]) + 1)
            self.cont_fit = True

        if 'res' in s[0]:
            self.res_num = max(self.res_num, int(s[0][4:]) + 1)

        if 'iso' in s[0]:
            self.add(s[0], addinfo=s[6])

        if 'me' in s[0]:
            if s[0] == 'me':
                s[0] = 'me_0'
            self.me_num = max(self.me_num, int(s[0][3:]) + 1)

        if 'cf' in s[0]:
            self.cf_fit = True
            attrs = ['val', 'left', 'right', 'step', 'vary', 'addinfo']
            self.parent.plot.add_pcRegion()

        if 'disp' in s[0]:
            self.disp_num = max(self.disp_num, int(s[0][6:]) + 1)

        if 'sts' in s[0]:
            self.stack_num = max(self.stack_num, int(s[0][4:]) + 1)

        for attr, val in zip(reversed(attrs), reversed(s[1:])):
            self.setValue(s[0], val, attr)
            if attr == 'val':
                self.setValue(s[0], float(s[4]), 'unc')
            if attr == 'addinfo' and 'cont_' in s[0]:
                self.cont[int(s[0].split('_')[1])].fromInfo(val)

        if 'cf' in s[0]:
            self.parent.plot.pcRegions[-1].updateFromFit()

    def showLines(self, sp=None):
        if sp is None:
            sp = list(set([s for sys in self.sys for s in sys.sp.keys()]))

        if len(sp) > 0:
            self.parent.console.exec_command('add ' + ' '.join(sp))

    def fromLMfit(self, result):
        for p in result.params.keys():
            par = result.params[p]
            name = str(par.name).replace('l4', '****').replace('l3', '***').replace('l2', '**').replace('l1', '*')
            self.setValue(name, self.pars()[name].ref(par.value), 'val')
            print(p, par.stderr)
            if isinstance(self.pars()[name].ref(par.stderr, attr='unc'), float):
                self.setValue(name, self.pars()[name].ref(par.stderr, attr='unc'), 'unc')
                self.setValue(name, self.pars()[name].ref(par.stderr, attr='unc'), 'step')
            else:
                self.setValue(name, 0, 'unc')

    def fromJulia(self, res, unc):
        s = ''
        for i, p in enumerate(self.list_fit()):
            self.setValue(p.__str__(), res[i])
            self.setValue(p.__str__(), unc[i], 'unc')
            self.setValue(p.__str__(), unc[i], 'step')
            #s += p.__str__() + ': ' + str(self.getValue(p.__str__(), 'unc')) + '\n'
            s += p.str(attr='lmfit') + '\n'
        return s

    def save(self):
        self.saved = OrderedDict()
        for k, v in self.pars().items():
            p = copy(v)
            for attr in ['val', 'min', 'max', 'step', 'addinfo']:
                setattr(p, attr, copy(getattr(v, attr)))
            self.saved[k] = p

    def load(self):
        for k in self.pars().keys():
            for attr in ['val', 'min', 'max', 'step', 'addinfo']:
                self.setValue(k, getattr(self.saved[k], attr), attr)

    def shake(self, scale=1):
        for p in self.list_fit():
            self.setValue(str(p), p.val + np.random.normal(scale=scale) * p.step, 'val')

    def setSpecific(self):
        self.addSys(z=2.8083543)
        self.setValue('b_0_SiIV', 10.6)
        self.setValue('N_0_SiIV', 12.71)
        self.addSys(z=2.8085)
        self.setValue('b_1_SiIV', 7.6)
        self.setValue('N_1_SiIV', 12.71)

    def __str__(self):
        return '\n'.join([str(s) for s in self.sys])

class line():
    def __init__(self, l0=1215.6682, f=2.776E-01, g=6.265E+08,name='HI'):
        self.l0 = l0
        self.f=f
        self.g=g
        self.name = name

class element():
    def __init__(self, N=10, b=10, z=0,name='tmp'):
        self.N=N
        self.b = b
        self.z = z
        self.lines = {}
        self.name = name

#class fit_model():
#    def __init__(self):


def read_model(filename='', zoom=True, skip_header=0):
    folder = os.path.dirname(filename)

    with open(filename) as f:
        d = f.readlines()

    i = -1 + skip_header
    spec_ind = -1
    spec = {}
    while (i < len(d) - 1):
        i += 1
        if '%' in d[i] or any([x in d[i] for x in ['spect', 'Bcont', 'fitting']]):
            if '%' in d[i]:
                specname = d[i][1:].strip()
                spec_ind+=1
                ind = -1
                i += 1
            else:
                ind = 0

            if i > len(d) - 1:
                break

            if ind == -1 and 'spectrum' in d[i]:
                n = int(d[i].split()[1])

                if n > 0:
                    x, y, err = [], [], []
                    if n > 0:
                        for t in range(n):
                            i += 1
                            w = d[i].split()
                            x.append(float(w[0]))
                            y.append(float(w[1]))
                            if len(w) > 2:
                                err.append(float(w[2]))
                    spec[spec_ind] = spectrum(np.asarray(x), np.asarray(y), np.asarray(err))
                    ind = 0

            if ind > -1:
                while all([x not in d[i] for x in ['%', '----', 'doublet', 'region', 'fit_model']]):

                    if 'fitting_points' in d[i]:

                        # self.s[ind].mask.set(x=np.zeros_like(self.s[ind].spec.x(), dtype=bool))
                        n = int(d[i].split()[1])
                        print('n fitting points:', n)
                        spec[spec_ind].fitting_points = []
                        if n > 0:
                            i += 1
                            for line in d[i:i + n]:
                                w = float(line.split()[0])
                                spec[spec_ind].fitting_points.append(w)
                                #w_pos = np.where(np.abs(spec[spec_ind].x-w)<1e-4)[0]
                                #if w_pos is not None:
                                #    spec[spec_ind].fitting_points[w_pos] = True

                    if 'resolution' in d[i]:
                        spec[spec_ind].resolution = int(float(d[i].split()[1]))

                    i += 1

                    if i > len(d) - 1:
                        break

        if '%' in d[i]:
            i -= 1

        if 'fit_model' in d[i]:
            fit = fitPars(parent=spec)
            num = int(d[i].split()[1])
            for k in range(num):
                i += 1
                fit.readPars(d[i])

    for s in spec.values():
        s.fit = fit
    return spec

if 0:
    species={}
    #add 12CI
    if 1:
        species['CI'] = element(name='CIj0')
        species['CI'].lines['CI1656'] = line(l0=1656.9284,f=1.49E-01,g=3.60E+08,name='CI')
        species['CI'].lines['CI1560'] = line(l0=1560.3092,f= 7.74E-02,g=1.27E+08,name='CI')
        species['CI'].lines['CI1328'] = line(l0=1328.8333,f=  7.58E-02 ,g=2.88E+08,name='CI')
        species['CI'].lines['CI1280'] = line(l0=1280.1352 ,f=  2.63E-02 ,g= 1.06E+08 ,name='CI')
        species['CI'].lines['CI1277'] = line(l0=1277.2452,f=   8.53E-02 ,g=2.32E+08,name='CI')
        species['CI'].lines['CI1276'] = line(l0=1276.4822,f=   5.89E-03 ,g=8.03E+06,name='CI')

        species['CI'].lines['CI1270'] = line(l0=1270.1432, f=3.86E-04, g=1.0E+08, name='CI')
        species['CI'].lines['CI1260'] = line(l0=1260.7351, f=5.07E-02, g=2.40E+08, name='CI')
        ################################################

        species['CIj1'] = element(name='CIj1')
        species['CIj1'].lines['CI1656.26'] = line(l0=1656.2672,f=6.21E-02,g=3.61E+08,name='CI')
        species['CIj1'].lines['CI1657.37'] = line(l0=1657.3792 ,f= 3.71E-02,g=3.60E+08,name='CI')
        species['CIj1'].lines['CI1657.90'] = line(l0=1657.9071,f=4.94E-02,g=3.60E+08,name='CI')
        species['CIj1'].lines['CI1560.6820'] = line(l0=1560.6820, f=5.81E-02,g=1.27E+08,name='CI')
        species['CIj1'].lines['CI1560.7089'] = line(l0=1560.7089 , f=1.93E-02,g=1.27E+08,name='CI')
        species['CIj1'].lines['CI1329.1233'] = line(l0=1329.1233,f=1.91E-02,g=2.88E+08 ,name='CI')
        species['CIj1'].lines['CI1329.1004'] = line(l0=1329.1004,f=3.13E-02,g=2.87E+08 ,name='CI')
        species['CIj1'].lines['CI1329.0849'] = line(l0=1329.0849,f=2.54E-02,g=2.89E+08,name='CI')
        species['CIj1'].lines['CI1287.6076'] = line(l0=1287.6076,f=6.03E-05,g=1.0E+08,name='CI')
        species['CIj1'].lines['CI1287.6076'] = line(l0=1287.6076,f=7.04E-03,g=1.05E+08,name='CI')
        species['CIj1'].lines['CI1280.4043'] = line(l0=1280.4043,f=4.40E-03,g=1.06E+08,name='CI')
        species['CIj1'].lines['CI1279.8907'] = line(l0=1279.8907 ,f=1.43E-02,g=1.17E+08,name='CI')
        species['CIj1'].lines['CI1279.0562'] = line(l0=1279.0562 ,f=7.08E-04,g=1.17E+08,name='CI')
        species['CIj1'].lines['CI1277.5131'] = line(l0=1277.5131 ,f=2.10E-02,g=2.32E+08,name='CI')
        species['CIj1'].lines['CI1277.2827'] = line(l0=1277.2827 ,f=6.66E-02,g=2.46E+08,name='CI')
        species['CIj1'].lines['CI1276.7498'] = line(l0=1276.7498 ,f=5.89E-03,g=1.0E+08,name='CI')

        species['CIj1'].lines['CI1270.408'] = line(l0=1270.408, f=4.51E-05 , g=1.0E+08, name='CI')
        species['CIj1'].lines['CI1276.7498'] = line(l0=1276.7498, f=5.89E-03, g=1.0E+08, name='CI')
        species['CIj1'].lines['CI1260.9262'] = line(l0=1260.9262 ,f=1.75E-02,g=2.41E+08,name='CI')
        species['CIj1'].lines['CI1260.9961'] = line(l0=1260.9961 ,f=1.34E-02,g=2.40E+08,name='CI')
        species['CIj1'].lines['CI1261.1224'] = line(l0=1261.1224, f=2.02E-02, g=2.36E+08, name='CI')
        ##############################################

        species['CIj2'] = element(name='CIj2')
        species['CIj2'].lines['CI1657.0081'] = line(l0=1657.00811,f=1.11E-01,g=3.61E+08,name='CI')
        species['CIj2'].lines['CI1657.9071'] = line(l0=1657.9071,f=3.71E-02,g=3.60E+08,name='CI')
        species['CIj2'].lines['CI1561.3399'] = line(l0=1561.3399,f=1.16E-02,g=1.27E+08,name='CI')
        species['CIj2'].lines['CI1561.3668'] = line(l0=1561.3668,f=7.72E-04,g=1.27E+08,name='CI')
        species['CIj2'].lines['CI1561.4378'] = line(l0=1561.4378,f=6.49E-02,g=1.27E+08,name='CI')
        species['CIj2'].lines['CI1329.6004'] = line(l0=1329.6004,f=1.89E-02,g=2.88E+08,name='CI')
        species['CIj2'].lines['CI1329.5775'] = line(l0=1329.5775,f=5.69E-02,g=2.87E+08,name='CI')
        species['CIj2'].lines['CI1288.0553'] = line(l0=1288.0553,f=2.01E-05,g=2.87E+08,name='CI')
        species['CIj2'].lines['CI1280.8471'] = line(l0=1280.8471,f=5.22E-03,g=1.06E+08,name='CI')
        species['CIj2'].lines['CI1280.3331'] = line(l0=1280.3331,f=1.52E-02,g=1.17E+08,name='CI')
        species['CIj2'].lines['CI1279.4980'] = line(l0=1279.4980,f=4.56E-04,g=1.0E+08,name='CI')
        species['CIj2'].lines['CI1279.2290'] = line(l0=1279.2290,f=2.14E-03,g=1.0E+08,name='CI')
        species['CIj2'].lines['CI1277.9539'] = line(l0=1277.9539,f=8.17E-04,g=2.32E+08,name='CI')
        species['CIj2'].lines['CI1277.7233'] = line(l0=1277.7233,f=1.53E-02,g=2.46E+08,name='CI')
        species['CIj2'].lines['CI1277.5501'] = line(l0=1277.5501,f=7.63E-02,g=2.44E+08,name='CI')
        species['CIj2'].lines['CI1277.1900'] = line(l0=1277.1900,f=3.22E-04 ,g=1.0E+08,name='CI')
        species['CIj2'].lines['CI1261.5519'] = line(l0=1261.5519, f=3.91E-02, g=2.36E+08, name='CI')
        species['CIj2'].lines['CI1261.4255'] = line(l0=1261.4255, f=1.31E-02, g=2.40E+08, name='CI')

    #add 13CI
    if 1:
        species['13CI'] = element(name='13CIj0')
        species['13CI'].lines['13CI1656.932'] = line(l0=1656.932,f=1.49E-01,g=3.60E+08,name='CI')
        species['13CI'].lines['13CI1560.292'] = line(l0=1560.292,f= 7.74E-02,g=1.27E+08,name='CI')
        species['13CI'].lines['13CI1328.826'] = line(l0=1328.826, f=7.58E-02, g=2.88E+08, name='CI')
        #species['13CI'].lines['13CI1280'] = line(l0=1280.1352, f=2.63E-02, g=1.06E+08, name='CI')
        species['13CI'].lines['13CI1277.2453'] = line(l0=1277.2453, f=8.53E-02, g=2.32E+08, name='CI')
        #species['13CI'].lines['13CI1276'] = line(l0=1276.4822, f=5.89E-03, g=8.03E+06, name='CI')
        species['13CI'].lines['13CI1260.7340'] = line(l0=1260.7340, f=5.07E-02, g=2.40E+08, name='CI')

        species['13CIj1'] = element(name='13CIj1')
        species['13CIj1'].lines['13CI1656.272'] = line(l0=1656.272,  f=6.21E-02, g=3.61E+08, name='CI')
        species['13CIj1'].lines['13CI1657.383'] = line(l0= 1657.383, f=3.71E-02, g=3.60E+08, name='CI')
        species['13CIj1'].lines['13CI1657.916'] = line(l0=1657.916 , f=4.94E-02, g=3.60E+08, name='CI')
        species['13CIj1'].lines['13CI1560.6644'] = line(l0=1560.6644, f=5.81E-02,g=1.27E+08,name='CI')
        species['13CIj1'].lines['13CI1560.6920'] = line(l0=1560.6920 , f=1.93E-02,g=1.27E+08,name='CI')
        species['13CIj1'].lines['13CI1329.116'] = line(l0=1329.116,f=1.91E-02,g=2.88E+08 ,name='CI')
        species['13CIj1'].lines['13CI1329.093'] = line(l0=1329.093,f=3.13E-02,g=2.87E+08 ,name='CI')
        species['13CIj1'].lines['13CI1329.079'] = line(l0=1329.079,f=2.54E-02,g=2.89E+08,name='CI')
        #species['CIj1'].lines['CI1287.6076'] = line(l0=1287.6076,f=6.03E-05,g=1.0E+08,name='CI')
        #species['CIj1'].lines['CI1287.6076'] = line(l0=1287.6076,f=7.04E-03,g=1.05E+08,name='CI')
        #species['CIj1'].lines['CI1280.4043'] = line(l0=1280.4043,f=4.40E-03,g=1.06E+08,name='CI')
        #species['CIj1'].lines['CI1279.8907'] = line(l0=1279.8907 ,f=1.43E-02,g=1.17E+08,name='CI')
        #species['CIj1'].lines['CI1279.0562'] = line(l0=1279.0562 ,f=7.08E-04,g=1.17E+08,name='CI')
        species['13CIj1'].lines['13CI1277.5134'] = line(l0=1277.5134 ,f=2.10E-02,g=2.32E+08,name='CI')
        species['13CIj1'].lines['13CI1277.2828'] = line(l0=1277.2828 ,f=6.66E-02,g=2.46E+08,name='CI')
        #species['CIj1'].lines['CI1276.7498'] = line(l0=1276.7498 ,f=5.89E-03,g=1.0E+08,name='CI')
        species['13CIj1'].lines['13CI1260.9254'] = line(l0=1260.9254 ,f=1.75E-02,g=2.41E+08,name='CI')
        species['13CIj1'].lines['13CI1260.9950'] = line(l0=1260.9950 ,f=1.34E-02,g=2.40E+08,name='CI')
        species['13CIj1'].lines['13CI1261.1213'] = line(l0=1261.1213, f=2.02E-02, g=2.36E+08, name='CI')

        species['13CIj2'] = element(name='CIj2')
        species['13CIj2'].lines['13CI1657.012'] = line(l0=1657.012, f=1.11E-01, g=3.61E+08, name='CI')
        species['13CIj2'].lines['13CI1658.125'] = line(l0=1658.125, f=3.71E-02, g=3.60E+08, name='CI')

        species['13CIj2'].lines['13CI1561.424'] = line(l0=1561.424, f=1.16E-02, g=1.27E+08, name='CI')
        species['13CIj2'].lines['13CI1561.350'] = line(l0=1561.350, f=7.72E-04, g=1.27E+08, name='CI')
        species['13CIj2'].lines['13CI1561.322 '] = line(l0=1561.322 , f=6.49E-02, g=1.27E+08, name='CI')
        species['13CIj2'].lines['13CI1329.593'] = line(l0=1329.593 , f=1.89E-02, g=2.88E+08, name='CI')
        species['13CIj2'].lines['CI1329.571'] = line(l0=1329.571, f=5.69E-02, g=2.87E+08, name='CI')
        #species['CIj2'].lines['CI1280.8471'] = line(l0=1280.8471, f=5.22E-03, g=1.06E+08, name='CI')
        #species['CIj2'].lines['CI1280.3331'] = line(l0=1280.3331, f=1.52E-02, g=1.17E+08, name='CI')
        #species['CIj2'].lines['CI1279.4980'] = line(l0=1279.4980, f=4.56E-04, g=1.0E+08, name='CI')
        #species['CIj2'].lines['CI1279.2290'] = line(l0=1279.2290, f=2.14E-03, g=1.0E+08, name='CI')

        species['13CIj2'].lines['13CI1277.9542'] = line(l0=1277.9542, f=8.17E-04, g=2.32E+08, name='CI')
        species['13CIj2'].lines['13CI1277.7234'] = line(l0=1277.7234, f=1.53E-02, g=2.46E+08, name='CI')
        species['13CIj2'].lines['13CI1277.5500'] = line(l0=1277.5500, f=7.63E-02, g=2.44E+08, name='CI')
        species['13CIj2'].lines['13CI1277.1906'] = line(l0=1277.1906, f=3.22E-04, g=1.0E+08, name='CI')

        species['13CIj2'].lines['13CI1261.5508'] = line(l0=1261.5508, f=3.91E-02, g=2.36E+08, name='CI')
        species['13CIj2'].lines['13CI1261.4244'] = line(l0=1261.4244, f= 1.31E-02, g=2.40E+08, name='CI')

    #add blends
    if 1:
        species['SiII'] = element(name='SiII')
        species['SiII'].lines['SiII1304.37'] = line(l0=1304.37020, f=8.63E-02, g=3.40E+08, name='SiII')
        species['SiII'].lines['SiII1526.70'] = line(l0=1526.707, f=1.33E-01, g=3.80E+08, name='SiII')

    with open('./species_database.pkl', 'wb') as f:
        pickle.dump(species, f)
else:
    with open('./species_database.pkl', 'rb') as f:
        species = pickle.load(f)

def plot_line(N=20,b=10,z=0,sp_line = line(),l=None):
    b *= 1.e5
    N = np.power(10,N)
    # Define line parameters, they are not important.
    l0 = sp_line.l0
    f = sp_line.f
    gam = sp_line.g

    if l is None:
        dl = 0.01
        l = np.arange(l0-10, l0+10, dl)

    profile = Voigt(l, l0, f, N, b, gam,z=z)
    #I = np.exp(-profile)
    if 0:
        plt.subplots()
        plt.plot(profile)
        plt.show()
    return profile



def calc_fit(model,plot_12C=True,plot_13C=True,verbose=False,plot_sys=-1):

    l_min,l_max = np.min(model.fitting_points)-1,np.max(model.fitting_points)+1
    step = model.fitting_points[0]/model.resolution/8
    if verbose:
        print(model.x[1]-model.x[0],model.x[0]/model.resolution,step)
    x = np.arange(l_min,l_max,step)
    f = np.zeros_like(x)
    
    iso = model.fit.iso.val
    for i_sys,s in enumerate(model.fit.sys):
        if plot_sys==-1:
            if verbose:
                print('sys:',i_sys)
            for sp in s.sp.values():
                if verbose:
                    print(sp.name,sp.N.val,sp.b.val,s.z.val)
                if sp.name in species.keys():
                    for l in species[sp.name].lines.values():
                        if verbose:
                            print(sp.name)
                            print(l.l0,l.f,l.g)
                        if plot_12C:
                            f += plot_line(N=sp.N.val,b=sp.b.val,z=s.z.val,sp_line = l,l=x)

                        if 0:
                            plt.subplots()
                            plt.plot(f)
                            plt.plot(plot_line(N=sp.N.val,b=sp.b.val,z=s.z.val,sp_line = l,l=x))
                            plt.show()
                    if iso is not None and '13' in sp.name:
                        if 0:
                            for l in species['13'+sp.name].lines.values():
                                if verbose:
                                    print('13'+sp.name)
                                    print(l.l0, l.f, l.g)
                                if plot_13C:
                                    f += plot_line(N=sp.N.val+iso, b=sp.b.val, z=s.z.val, sp_line=l, l=x)
                        else:
                            for l in species[sp.name].lines.values():
                                if verbose:
                                    print(sp.name)
                                    print(l.l0, l.f, l.g)
                                if plot_13C:
                                    f += plot_line(N=sp.N.val, b=sp.b.val, z=s.z.val, sp_line=l, l=x)
        elif i_sys == plot_sys:
            if verbose:
                print('sys:', i_sys)
            for sp in s.sp.values():
                if verbose:
                    print(sp.name, sp.N.val, sp.b.val, s.z.val)
                if sp.name in species.keys():
                    for l in species[sp.name].lines.values():
                        if verbose:
                            print(sp.name)
                            print(l.l0, l.f, l.g)
                        if plot_12C:
                            f += plot_line(N=sp.N.val, b=sp.b.val, z=s.z.val, sp_line=l, l=x)

                        if 0:
                            plt.subplots()
                            plt.plot(f)
                            plt.plot(plot_line(N=sp.N.val, b=sp.b.val, z=s.z.val, sp_line=l, l=x))
                            plt.show()
                    if iso is not None and '13' in sp.name:
                        if 0:
                            for l in species['13' + sp.name].lines.values():
                                if verbose:
                                    print('13' + sp.name)
                                    print(l.l0, l.f, l.g)
                                if plot_13C:
                                    f += plot_line(N=sp.N.val + iso, b=sp.b.val, z=s.z.val, sp_line=l, l=x)
                        else:
                            for l in species[sp.name].lines.values():
                                if verbose:
                                    print(sp.name)
                                    print(l.l0, l.f, l.g)
                                if plot_13C:
                                    f += plot_line(N=sp.N.val, b=sp.b.val, z=s.z.val, sp_line=l, l=x)
    return x,f


def find_fitting_points(x,lst):
    mask = x<0
    pix_size = np.median(np.diff(x))
    for l in lst:
        #mask_pos = np.where(np.abs(x-l)<pix_size)[0]
        mask_pos = np.where(x<=l)[0][-1]
        mask[mask_pos] = True

    return mask

if __name__ == '__main__':

    # set figsize and fontsize
    fig_size = (12,6)
    fontsize = 10


    q_name = 'J0016'
    #create case for the quasar
    if q_name == 'J0016':
        #set the path to espresso normalized spectrum
        espresso_sp = np.loadtxt('/home/slava/science/research/kulkarni/C_isotopes/joint_fit/analysis/J0016/spec_es.dat')
        #set the path to uves normalized spectrum
        uves_sp = np.loadtxt('/home/slava/science/research/kulkarni/C_isotopes/joint_fit/analysis/J0016/spec_uves.dat')
        #set the path to model spv file
        model = read_model(filename='/home/slava/science/research/kulkarni/C_isotopes/joint_fit/analysis/J0016/joined_model.spv')
        #set the x and y ranges
        lmin, lmax,deltal = -200,200,50
        fmin, fmax,deltaf = -0.1, 1.3,0.5
        #set figname
        figname = 'J0016'


    fig,ax = plt.subplots(3,2,figsize=fig_size)
    fig.subplots_adjust(wspace=0.15,hspace=0.2)

    #calc fits for CI
    if 1:
        uves_fit = {}
        uves_resolution = 46000  # UVES
        x_13,f_13 = calc_fit(model[0],plot_12C=False)
        fit_13 = convolve_res2(x_13, np.exp(-f_13), uves_resolution)
        uves_fit['C13'] = spectrum(x_13,fit_13)

        x_12,f_12 = calc_fit(model[0],plot_13C=False)
        fit_12 = convolve_res2(x_12, np.exp(-f_12), uves_resolution)
        uves_fit['C12'] = spectrum(x_12,fit_12)

        x_tot, f_tot = calc_fit(model[0], plot_13C=True,plot_12C=True)
        fit_tot = convolve_res2(x_tot, np.exp(-f_tot), uves_resolution)
        uves_fit['Ctot'] = spectrum(x_tot,fit_tot)

        #calc_sys
        nsys = len(model[0].fit.sys)
        for i in range(nsys):
            x,y =  calc_fit(model[0],plot_13C=False,plot_sys=i)
            y = convolve_res2(x, np.exp(-y), uves_resolution)
            uves_fit['C12_'+str(i)] = spectrum(x,y)


        espresso_fit = {}
        espresso_resolution = 140000# ESPRESSO
        x_13,f_13 = calc_fit(model[1],plot_12C=False)
        fit_13 = convolve_res2(x_13, np.exp(-f_13), espresso_resolution)
        espresso_fit['C13'] = spectrum(x_13,fit_13)

        x_12,f_12 = calc_fit(model[1],plot_13C=False)
        fit_12 = convolve_res2(x_12, np.exp(-f_12), espresso_resolution)
        espresso_fit['C12'] = spectrum(x_12,fit_12)

        x_tot, f_tot = calc_fit(model[1], plot_13C=True,plot_12C=True)
        fit_tot = convolve_res2(x_tot, np.exp(-f_tot), espresso_resolution)
        espresso_fit['Ctot'] = spectrum(x_tot,fit_tot)





        def plot_panel(axs=ax[0,0],l0=1656,instr='UVES',z_abs=0,x_coord_type='velocity',
                       plot_residuals=False, plot_components= False,legend=False):
            if x_coord_type=='velocity':
                def f_x(x):
                    vc = 299792
                    return vc*(x/(l0*(1+z_abs))-1)
                xlow,xup = -1000,1000
            elif x_coord_type=='wavelength':
                def f_x(x):
                    return x /(1+z_abs)
            if instr=='UVES':
                f = uves_fit
                s = uves_sp
                m = model[0]
                fitting_points = find_fitting_points(x=s[:,0],lst=m.fitting_points)
            if instr == 'ESPRESSO':
                f = espresso_fit
                s = espresso_sp
                m = model[1]
                fitting_points = find_fitting_points(x=s[:, 0], lst=m.fitting_points)

            axs.step(f_x(s[:,0]),s[:,1],where='mid',color='grey',zorder=-10,lw=0.5)
            y = np.array(s[:,1])
            y[~fitting_points] = np.nan
            axs.errorbar(x=f_x(s[:, 0]), y=y, ls='-', ds='steps-mid', color='black', lw=0.5,zorder=-1)
            del(y)

            #axs.plot(f_x(f['C12'].x),f['C12'].y,color='blue',lw=1)
            label = '_nolegend_'
            if legend:
                label =('C13')
            axs.plot(f_x(f['C13'].x),f['C13'].y,color='darkorange',lw=1.5,label=label)
            if legend:
                label =('Ctot')
            axs.plot(f_x(f['Ctot'].x),f['Ctot'].y,color='red',lw=1,label=label)

            if plot_components:
                label = '_nolegend_'
                for i in range(len(m.fit.sys)):
                    if legend and i == len(m.fit.sys)-1:
                        label = ('components')
                    axs.plot(f_x(f['C12_'+str(i)].x), f['C12_'+str(i)].y, color='blue', lw=0.5,label=label)

            if plot_residuals and 0:
                fit_interp = interp1d(f_x(f['Ctot'].x), f['Ctot'].y,fill_value='extrapolate')
                spec_interp = interp1d(f_x(s[:,0]), s[:,1],fill_value='extrapolate')
                err_interp = interp1d(f_x(s[:,0]), s[:,2],fill_value='extrapolate')

                x_tmp = f_x(m.x[m.fitting_points])
                x_residuals =x_tmp[(x_tmp>xlow)*(x_tmp<xup)]
                del x_tmp
                #xx = f_x(s[:,0])
                y_residuals = (spec_interp(x_residuals)-fit_interp(x_residuals))/err_interp(x_residuals)
                axs.plot(x_residuals,2+y_residuals*0.1,'.',color='black')
                axs.axhline(2.1,color='black',lw=1)
                axs.axhline(1.9,color='black',lw=1)
                fmin,fmax = -0.1, 2.3


        z_abs = model[0].fit.sys[0].z.val
        plot_panel(axs=ax[0, 0], l0=species['CI'].lines['CI1656'].l0, instr='UVES', z_abs=z_abs,
                   x_coord_type='velocity', plot_components= True,legend=True)
        plot_panel(axs=ax[1, 0], l0=species['CI'].lines['CI1560'].l0, instr='UVES', z_abs=z_abs,
                   x_coord_type='velocity', plot_components= True,legend=True)
        plot_panel(axs=ax[2, 0], l0=species['CI'].lines['CI1328'].l0, instr='UVES', z_abs=z_abs,
                   x_coord_type='velocity', plot_components= True,legend=True)
        #plot_panel(axs=ax[3, 0], l0=species['CI'].lines['CI1277'].l0, instr='UVES', z_abs=z_abs,
        #           x_coord_type='velocity')



        z_abs = model[0].fit.sys[0].z.val
        plot_panel(axs=ax[0, 1], l0=species['CI'].lines['CI1656'].l0, instr='ESPRESSO', z_abs=z_abs,
                   x_coord_type='velocity')
        plot_panel(axs=ax[1, 1], l0=species['CI'].lines['CI1560'].l0, instr='ESPRESSO', z_abs=z_abs,
                   x_coord_type='velocity')
        plot_panel(axs=ax[2, 1], l0=species['CI'].lines['CI1328'].l0, instr='ESPRESSO', z_abs=z_abs,
                   x_coord_type='velocity')
        #plot_panel(axs=ax[3, 1], l0=species['CI'].lines['CI1277'].l0, instr='ESPRESSO', z_abs=z_abs,
        #           x_coord_type='velocity')

        for i in range(2):
            for axs in ax[:,i]:
                axs.axhline(0,ls=':',lw=1)
                axs.set_xlabel('v, km/s',fontsize=fontsize)
                axs.set_ylabel('Flux',fontsize=fontsize)
                axs.set_xlim(lmin, lmax)
                axs.set_ylim(fmin, fmax)
                axs.xaxis.set_minor_locator(AutoMinorLocator(5))
                axs.xaxis.set_major_locator(MultipleLocator(deltal))
                axs.yaxis.set_minor_locator(AutoMinorLocator(5))
                axs.yaxis.set_major_locator(MultipleLocator(deltaf))
                axs.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True',
                                top='True')
                axs.tick_params(which='major', length=5)
                axs.tick_params(which='minor', length=3)

        labels = ['CI 1656','CI 1560','CI 1328']
        for i in range(3):
            xl,xu = axs.get_xlim()
            yl,yu = axs.get_ylim()
            ax[i,0].text(xl+0.05*(xu-xl),yl+0.1*(yu-yl),labels[i],fontsize=fontsize)
            ax[i, 0].legend(loc='lower right',fontsize=fontsize-2)

        ax[0,0].set_title('UVES')
        ax[0,1].set_title('ESPRESSO')

        fig.savefig(figname+'.pdf', bbox_inches='tight')

        plt.show()
