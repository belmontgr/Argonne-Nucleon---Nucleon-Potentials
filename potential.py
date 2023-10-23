import numpy as np
import matplotlib.pyplot as plt


def V1S0(r, lpot):
    l = 0
    s = 0
    j = 0
    t = 1
    t1z = -1
    t2z = -1
    vpw = av18pw(lpot, l, s, j, t, t1z, t2z, r)
    return vpw[0][0]

# **********************************************************************
# Argonne v18 and vn' and Super-Soft Core (C) potential package 
#
# prepared 1 Sep 1994 by R. B. Wiringa, Physics Division, 
# Argonne National Laboratory, Argonne, IL 60439
# e-mail: wiringa@theory.phy.anl.gov 
# 
# reference: 
#  "Accurate nucleon-nucleon potential with charge-independence breaking"
#   R. B. Wiringa, V. G. J. Stoks, and R. Schiavilla, 
#   Physical Review C51, 38 (1995) - WSS95.
#
# option for v8' reprojection of v18 potential added 10 Jan 2001
#
# reference:
#  "Quantum Monte Carlo calculations of nuclei with A<=7"
#   B. S. Pudliner, V. R. Pandharipande, J. Carlson, Steven C. Pieper,
#   and R. B. Wiringa, Physical Review C56, 1720 (1997) - PPCPW97
#
# option for v6', v4', vx', v2', v1' potentials added 16 Jul 2002
#
# reference:
#  "Evolution of Nuclear Spectra with Nuclear Forces"
#   R. B. Wiringa and Steven C. Pieper,
#   Physical Review Letters 89, 182501 (2002) - WP02
#
# option for Super-Soft Core (C) added 14 Feb 2007
#
# reference:
#  "Construction d'un potentiel nucleon-nucleon a coeur tres mou (SSC)"
#   R. de Tourreil and D. W. L. Sprung,
#   Nuclear Physics A201, 193 (1973) - TS73
#
# option for modfied Argonne v8' and modified SSCC v8' added 14 Feb 2007
#   modifications selected by Steve Pieper
#   correction to modified SSCC v8' on 4 Apr 2007
#
# this file contains 4 subroutines:
#   subroutine av18pw(lpot,l,s,j,t,t1z,t2z,r,vpw)
#   subroutine av18op(lpot,r,vnn)
#   subroutine empot(lpot,r,vem)
#   subroutine consts(lpot,hc,mpi0,mpic,mp,mn,alpha,mup,mun)
#
# the variable lpot selects between v18, v8' and other options
#
# av18pw gives the full potential in a particular partial wave
# av18op gives the strong interaction part in operator format
# empot  gives the electromagnetic part in operator format
# consts gives values of fundamental constants and masses used
#
# notes: 
# 1) av18pw includes full EM interaction for lpot=1;
#    for lpot>1 it includes only C1(pp), i.e.,
#    Coulomb with a form factor for pp channels.
#
# 2) empot does not include the energy-dependence of the Coulomb
#    interaction used in eq.(4) of WSS95, i.e., it uses alpha, 
#    not alpha'.
#
# 3) the vacuum polarization in empot is a short-range approximation 
#    to eq.(7) suitable for bound states, but not for scattering.
#    it is taken from eq.(3.13) of Rev. Mod. Phys. 44, 48 (1972)
#
#    8/28/97 error in this formula detected and corrected:
#    should be -(gamma+5/6) instead of printed (-gamma+5/6)
#
# 4) these subroutines should be compiled with a compiler option
#    that forces all floating point constants to be evaluated at
#    real*8 significance, e.g., on an IBM RS6000 the xlf compiler
#    option qdpc=e should be used; on SGI machines, the -r8 option
#    should be used; on a Cray no action is needed.
#    if such an option is not available and the default precision is 
#    real*4 (32 bits), then all constants should be explicitly 
#    converted to double precision by appending a D0.
#
# 5) consts now (14 Feb 2007) depend upon potential:
#    need to call to generate appropriate hbar**2/M
#
# *id* av18pw **********************************************************
# subroutine for partial-wave projection of argonne v18 potential
# or super-soft core (C) v14 potential
# or reprojected vn' potential
# calls subroutines av18op, empot
# ----------------------------------------------------------------------
# arguments for av18pw
# lpot: switch for potential choice
#     -----------------------------------------------
#         Argonne                Super-Soft Core (C)
#       = 1 : av18              = 101 : sscc v14
#       = 2 : av8'              = 102 : sscc v8'
#       = 3 : av6'
#       = 4 : av4'
#       = 5 : avx'
#       = 6 : av2'
#       = 7 : av1'
#       = 8 : modified av8'     = 108 : modified sscc v8'
#     -----------------------------------------------
# l:    orbital angular momentum of pair (0,1,2,...)
# s:    total spin of pair (0 or 1)
# j:    total angular momentum of pair (0,1,2,...)
# t:    total isospin of pair (0 or 1)
# t1z:  isospin of particle 1 (1 for p, -1 for n)
# t2z:     "    "     "     2 (1 for p, -1 for n)
# r:    separation in fm
# v:    returned potential in MeV (2x2 array)
#       (includes all strong and em terms)
# ----------------------------------------------------------------------
# order of terms in v(l,m):
#      single channel                 coupled channel (l=j-1,s=1)
#      v(1,1) = v(l,s,j,t,t1z,t2z)    v(1,1) = v(l,s,j,t,t1z,t2z)
#      v(2,1) = 0                     v(2,1) = v(l<->l+2)
#      v(1,2) = 0                     v(1,2) = v(l<->l+2)
#      v(2,2) = 0                     v(2,2) = v(l+2,s,j,t,t1z,t2z)
# ----------------------------------------------------------------------
def av18pw(lpot, l, s, j, t, t1z, t2z, r):
    vnn = [0]*18
    vem = [0]*14
    vpw = [[0, 0], [0, 0]]
    
    # Strong interaction terms
    vnn = av18op(lpot, r)
    
    s1ds2 = 4 * s - 3
    t1dt2 = 4 * t - 3
    t12 = 3 * t1z * t2z - t1dt2
    
    vc = vnn[0] + t1dt2 * vnn[1] + s1ds2 * vnn[2] + s1ds2 * t1dt2 * vnn[3] + t12 * vnn[14] + s1ds2 * t12 * vnn[15] + (t1z + t2z) * vnn[17]
         
    vt = vnn[4] + t1dt2 * vnn[5] + t12 * vnn[16]
    vls = vnn[6] + t1dt2 * vnn[7]
    vl2 = vnn[8] + t1dt2 * vnn[9] + s1ds2 * vnn[10] + s1ds2 * t1dt2 * vnn[11]
    vls2 = vnn[12] + t1dt2 * vnn[13]
    
    # Electromagnetic terms
    vem = empot(lpot, r)
    
    if t1z + t2z < 0:
        vc += s1ds2 * vem[6]
        vt += vem[9]
    elif t1z + t2z == 0:
        vc += vem[4] + s1ds2 * vem[7]
        vt += vem[10]
        vls += vem[13]
    else:
        vc += sum(vem[:4]) + s1ds2 * vem[5]
        vt += vem[8]
        vls += vem[11]
        
    ncc = 1
    if s == 1 and j > l:
        ncc = 2
    
    if ncc == 1:
        s12 = 0
        if s == 1 and l == j:
            s12 = 2
        if l == (j + 1):
            s12 = -2 * (j + 2) / (2 * j + 1)
        ls = (j * (j + 1) - l * (l + 1) - s * (s + 1)) / 2
        vpw[0][0] = vc + s12 * vt + ls * vls + l * (l + 1) * vl2 + ls**2 * vls2
    else:
        s12m = -2 * (j - 1) / (2 * j + 1)
        s12 = (6 * j * (j + 1))**(0.5) / (2 * j + 1)
        s12p = -2 * (j + 2) / (2 * j + 1)
        lsm = j - 1
        lsp = -(j + 2)
        vpw[0][0] = vc + s12m * vt + lsm * vls + l * (l + 1) * vl2 + lsm**2 * vls2
        vpw[1][0] = s12 * vt
        vpw[0][1] = s12 * vt
        vpw[1][1] = vc + s12p * vt + lsp * vls + (l + 2) * (l + 3) * vl2 + lsp**2 * vls2
        
    return vpw

# *id* av18op **********************************************************
# subroutine for strong interaction part of argonne v18 potential 
# or super-soft core (C) v14 potential
# or reprojected vn' potential in operator format
# calls subroutine consts
# ----------------------------------------------------------------------
# arguments for av18pot
# lpot: switch for potential choice
#     -----------------------------------------------
#         Argonne                Super-Soft Core (C)
#       = 1 : av18              = 101 : sscc v14
#       = 2 : av8'              = 102 : sscc v8'
#       = 3 : av6'
#       = 4 : av4'
#       = 5 : avx'
#       = 6 : av2'
#       = 7 : av1'
#       = 8 : modified av8'     = 108 : modified sscc v8'
#     -----------------------------------------------
# r:    separation in fm
# vnn:  output potential in MeV (18 component array)
# ----------------------------------------------------------------------
# order of operators l in vnn(l):
# l:    1=1                              2=t1.t2
#       3=s1.s2                          4=(s1.s2)(t1.t2)
#       5=S12 [=3(s1.r)(s2.r)-s1.s2]     6=S12(t1.t2)
#       7=L.S                            8=L.S(t1.t2)
#       9=L**2                          10=L**2(t1.t2)
#      11=L**2(s1.s2)                   12=L**2(s1.s2)(t1.t2)
#      13=(L.S)**2                      14=(L.S)**2(t1.t2)
#      15=T12 [=3*t1z*t2z-t1.t2]        16=(s1.s2)T12
#      17=S12*T12                       18=t1z+t2z
# where s1=sigma_1, t1=tau_1, t1z=tau_1(z), etc.
# ----------------------------------------------------------------------
def av18op(lpot, r):
    vnn = np.zeros(18)  # Initialize the array with zeros
    small = 1e-4
    vsmall = 1e-10

    # Define the functions
    def yc(t):
        return np.exp(-t) / x

    def yt(t):
        return (1 + 3 / t + 3 / t**2) * np.exp(-t) / x

    def yls(t):
        return -(1 + t) * np.exp(-t) / x**3

    def yl2(t):
        return (1 + 2 / t) * np.exp(-t) / x**3

    # argonne potential and derivatives
    if lpot < 100:
        hc, mpi0, mpic, mp, mn, alpha, mup, mun = consts(lpot) # The consts function has to be defined elsewhere
        mpi = (mpi0 + 2. * mpic) / 3.
        mu0 = mpi0 / hc
        muc = mpic / hc
        mu = mpi / hc
        fsq = .075
        cpi = 2.1
        rws = .5
        aiws = 5.
        x = mu * r
        x0 = mu0 * r
        xc = muc * r

        if r <= small:
            tpi = 3 * cpi**2 * r / mu**3
            ypi0 = (mpi0 / mpic)**2 * (mpi0 / 3) * cpi * r / mu0
            tpi0 = 3 * cpi * ypi0 / mu0**2
            ypic = (mpic / 3) * cpi * r / muc
            tpic = 3 * cpi * ypic / muc**2
        else:
            rcut = 1 - np.exp(-cpi * r * r)
            ypi = np.exp(-x) * rcut / x
            tpi = (1 + (3 + 3 / x) / x) * ypi * rcut
            ypi0 = (mpi0 / mpic)**2 * (mpi0 / 3) * np.exp(-x0) * rcut / x0
            tpi0 = (1 + (3 + 3 / x0) / x0) * ypi0 * rcut
            ypic = (mpic / 3) * np.exp(-xc) * rcut / xc
            tpic = (1 + (3 + 3 / xc) / xc) * ypic * rcut 
            ypi0 = fsq * ypi0
            ypic = fsq * ypic
        tpi0 = fsq * tpi0
        tpic = fsq * tpic
        tpi2 = tpi * tpi
        ws = 1 / (1 + np.exp((r - rws) * aiws))
        ws0 = 1 / (1 + np.exp(-rws * aiws))
        wsp = ws * (1 + aiws * np.exp(-rws * aiws) * ws0 * r)
        wsx = ws * x
        wsx2 = wsx * x
        dypi00 = (mpi0 / mpic)**2 * (mpi0 / 3) * cpi / mu0
        dypic0 = (mpic / 3) * cpi / muc
        ypi0p = ypi0 - fsq * dypi00 * ws * r / ws0
        ypicp = ypic - fsq * dypic0 * ws * r / ws0
        ypi = (ypi0 + 2 * ypic) / 3
        tpi = (tpi0 + 2 * tpic) / 3
        p11pp = -7.62701 * tpi2 + 1815.4920 * wsp + 1847.8059 * wsx2 + ypi0p
        p11np = -7.62701 * tpi2 + 1813.5315 * wsp + 1847.8059 * wsx2 - ypi0p + 2 * ypicp
        p11nn = -7.62701 * tpi2 + 1811.5710 * wsp + 1847.8059 * wsx2 + ypi0p
        pt1pp = 1.07985 * tpi2 - 190.0949 * wsx - 811.2040 * wsx2 + tpi0
        pt1np = 1.07985 * tpi2 - 190.0949 * wsx - 811.2040 * wsx2 - tpi0 + 2 * tpic
        pt1nn = 1.07985 * tpi2 - 190.0949 * wsx - 811.2040 * wsx2 + tpi0
        pls1 = -0.62697 * tpi2 - 570.5571 * wsp + 819.1222 * wsx2
        pl211 = 0.06709 * tpi2 + 342.0669 * wsp - 615.2339 * wsx2
        pls21 = 0.74129 * tpi2 + 9.3418 * wsp - 376.4384 * wsx2
        p10 = -8.62770 * tpi2 + 2605.2682 * wsp + 441.9733 * wsx2 - ypi0p - 2 * ypicp
        pt0 = 1.485601 * tpi2 - 1126.8359 * wsx + 370.1324 * wsx2 - tpi0 - 2 * tpic
        pls0 = 0.10180 * tpi2 + 86.0658 * wsp - 356.5175 * wsx2
        pl210 = -0.13201 * tpi2 + 253.4350 * wsp - 1.0076 * wsx2
        pls20 = 0.07357 * tpi2 - 217.5791 * wsp + 18.3935 * wsx2
        p01pp = -11.27028 * tpi2 + 3346.6874 * wsp - 3 * ypi0p
        p01np = -10.66788 * tpi2 + 3126.5542 * wsp - 3 * (-ypi0p + 2 * ypicp)
        p01nn = -11.27028 * tpi2 + 3342.7664 * wsp - 3 * ypi0p
        pl201 = 0.12472 * tpi2 + 16.7780 * wsp
        p00 = -2.09971 * tpi2 + 1204.4301 * wsp - 3 * (-ypi0p - 2 * ypicp)
        pl200 = -0.31452 * tpi2 + 217.4559 * wsp
        p11 = (p11pp + p11nn + p11np) / 3
        p11cd = (0.5 * (p11pp + p11nn) - p11np) / 6
        p11cs = (p11pp - p11nn) / 4
        pt1 = (pt1pp + pt1nn + pt1np) / 3
        pt1cd = (0.5 * (pt1pp + pt1nn) - pt1np) / 6
        pt1cs = (pt1pp - pt1nn) / 4
        p01 = (p01pp + p01nn + p01np) / 3
        p01cd = (0.5 * (p01pp + p01nn) - p01np) / 6
        p01cs = (p01pp - p01nn) / 4


        if lpot >= 2:
            p00 = p00 + 2 * pl200
            p11 = p11 + 2 * pl211 + 4 * pls21 / 3
            pt1 = pt1 - 5 * pls21 / 12
            pls1 = pls1 - 0.5 * pls21
            pls0 = pls0 - 2 * pl210 - 3 * pls20
        # option for v6' reduction
        if 3 <= lpot <= 7:
            p10 = p10 - 0.3 * pls0
        # option for v4' reduction
        if 4 <= lpot <= 7:
            p10 = p10 + 0.8735 * pt0
        # option for vx' reduction
        if lpot == 5:
            vnn[0] = 0.0625 * (9 * p11 + 3 * p10 + 3 * p01 + p00)
            vnn[1] = 0.0125 * (9 * p11 - 5 * p10 - 5 * p01 + p00)
            vnn[2] = vnn[1]
            vnn[3] = vnn[1]
            return vnn
        elif lpot == 6:
            vnn[0] = 0.25 * (3 * p01 + p10)
            vnn[1] = 0.25 * (p01 - p10)
            return vnn
        # option for v1' reduction
        elif lpot == 7:
            vnn[0] = 0.5 * (p01 + p10)
            return vnn
        # option for modified v8'
        if lpot == 8:
            p11 = p11 - 0.37 * tpi2
        vnn[0] = 0.0625 * (9 * p11 + 3 * p10 + 3 * p01 + p00)
        vnn[1] = 0.0625 * (3 * p11 - 3 * p10 + p01 - p00)
        vnn[2] = 0.0625 * (3 * p11 + p10 - 3 * p01 - p00)
        vnn[3] = 0.0625 * (p11 - p10 - p01 + p00)
        if lpot == 4:
           return vnn

        vnn[4] = 0.25 * (3 * pt1 + pt0)
        vnn[5] = 0.25 * (pt1 - pt0)
        if lpot == 3:
            return vnn

        vnn[6] = 0.25 * (3 * pls1 + pls0)
        vnn[7] = 0.25 * (pls1 - pls0)
        if lpot == 2 or lpot == 8:
            return vnn
        vnn[8] = 0.0625 * (9 * pl211 + 3 * pl210 + 3 * pl201 + pl200)
        vnn[9] = 0.0625 * (3 * pl211 - 3 * pl210 + pl201 - pl200)
        vnn[10] = 0.0625 * (3 * pl211 + pl210 - 3 * pl201 - pl200)
        vnn[11] = 0.0625 * (pl211 - pl210 - pl201 + pl200)
        vnn[12] = 0.25 * (3 * pls21 + pls20)
        vnn[13] = 0.25 * (pls21 - pls20)
        vnn[14] = 0.25 * (3 * p11cd + p01cd)
        vnn[15] = 0.25 * (p11cd - p01cd)
        vnn[16] = pt1cd
        vnn[17] = p01cs
        return vnn
    # super-soft core (C) potential and derivatives
    elif lpot > 100:
        if r <= vsmall:
            r = vsmall
        x = 0.7 * r
        rr4 = r**4
        rc4 = 1 - np.exp(-rr4)
        rc6 = 1 - np.exp(-r**6)
        hr = 10.463
        p11 = 144.83 * np.exp(-rr4 / 0.88787**2) + (-241.34 * yc(3.3788 * x) + (hr / 3) * yc(x)) * rc4
        p10 = 215.32 * np.exp(-rr4 / 0.85807**2) + (-883.6 * yc(3.5042 * x) - hr * yc(x)) * rc4
        p01 = 375 * np.exp(-rr4 / 0.47552**2) + (-1001.6 * yc(3.6071 * x) - hr * yc(x)) * rc4
        p00 = 75.653 * np.exp(-rr4 / 3**2) + (-286.26 * yc(2.0254 * x) + 3 * hr * yc(x)) * rc4
        pt1 = 36 * np.exp(-rr4 / 1.0805**2) + (-110 * yt(3.9529 * x) + (hr / 3) * yt(x)) * rc6
        pt0 = -58.951 * np.exp(-rr4 / 1.3171**2) + (395.18 * yt(4.3098 * x) - hr * yt(x)) * rc6
        pls1 = (520 * yls(5.661 * x) - 54.85 * yls(4.0141 * x)) * rc6
        pls0 = (-40.466 * yls(5.768 * x) - 40.408 * yls(4.0676 * x)) * rc6
        pl211 = (6.65 * yl2(1.965 * x) - 0.959 * yl2(x)) * rc6
        pl210 = (17.626 * yl2(2.6463 * x) - 0.35261 * yl2(x)) * rc6
        pl201 = (14 * yl2(2.5 * x) - 0.35 * yl2(x)) * rc6
        pl200 = (15.633 * yl2(2.01 * x) + 0.72581 * yl2(x)) * rc6
        pq0 = -3.9904 * yl2(2.4583 * x) * rc6
        # option for v8' reduction
    if lpot >= 102:
        p00 = p00 + 2 * pl200
        pls0 = pls0 - 2 * pl210 - 10 * pq0
        p11 = p11 + 2 * pl211
# option for modified v8'
    if lpot == 108:
        p11 = p11 - 111 * yc(3.3788 * x) * rc4
    vnn[0] = 0.0625 * (9 * p11 + 3 * p10 + 3 * p01 + p00)
    vnn[1] = 0.0625 * (3 * p11 - 3 * p10 + p01 - p00)
    vnn[2] = 0.0625 * (3 * p11 + p10 - 3 * p01 - p00)
    vnn[3] = 0.0625 * (p11 - p10 - p01 + p00)
    vnn[4] = 0.25 * (3 * pt1 + pt0)
    vnn[5] = 0.25 * (pt1 - pt0)
    vnn[6] = 0.25 * (3 * pls1 + pls0) + 0.75 * pq0
    vnn[7] = 0.25 * (pls1 - pls0) - 0.75 * pq0
    if lpot >= 102:
        return vnn
    vnn[8] = 0.0625 * (9 * pl211 + 3 * pl210 + 3 * pl201 + pl200) - 0.75 * pq0
    vnn[9] = 0.0625 * (3 * pl211 - 3 * pl210 + pl201 - pl200) + 0.75 * pq0
    vnn[10] = 0.0625 * (3 * pl211 + pl210 - 3 * pl201 - pl200) - 0.25 * pq0
    vnn[11] = 0.0625 * (pl211 - pl210 - pl201 + pl200) + 0.25 * pq0
    vnn[12] = 1.5 * pq0
    vnn[13] = -1.5 * pq0
    return vnn


# *id* empot ***********************************************************
# subroutine for electromagnetic part of Argonne v18 potential
# for avn' models returns pp Coulomb only
# calls subroutine consts
# ----------------------------------------------------------------------
# arguments for empot
# lpot: switch for potential choice
#       = 1 : full EM potential
#       > 1 : C1(pp) only
# r:    input separation in fm
# vem:  output potential in MeV (14 component array)
# ----------------------------------------------------------------------
# order of operators in vem(l)
# l:    1=C1    (pp)          2=DF    (pp)          3=C2      (pp)
#       4=VP    (pp)                                5=C1      (np)
#       6=s1.s2 (pp)          7=s1.s2 (nn)          8=s1.s2   (np)
#       9=S12   (pp)         10=S12   (nn)         11=S12     (np)
#      12=L.S   (pp)         13=L.S   (nn)         14=L.S     (np)
# C1 = one-photon-exchange Coulomb with form factor
# C2 = two-photon-exchange Coulomb
# DF = Darwin-Foldy
# VP = vacuum polarization (short-range approximation)
# all other terms from magnetic moment (MM) interactions
# ----------------------------------------------------------------------
def empot(lpot, r):
    small = 1e-5
    vem = np.zeros(14)

    # Call to consts function, you'll need to provide its Python implementation
    hc, mpi0, mpic, mp, mn, alpha, mup, mun = consts(lpot)
    
    b = 4.27
    br = b * r
    pi = np.pi
    me = 0.510999
    mr = mp * mn / (mp + mn)
    gamma = 0.577216
    beta = .0189

    if r < small:
        fcoulr = 5 * b / 16
        ftr3 = b ** 3 * br ** 2 / 720
        flsr3 = b ** 3 / 48
        kr = me * small / hc
    else:
        fcoulr = (1 - (1 + 11 * br / 16 + 3 * br ** 2 / 16 + br ** 3 / 48) * np.exp(-br)) / r
        ftr3 = (1 - (1 + br + br ** 2 / 2 + br ** 3 / 6 + br ** 4 / 24 + br ** 5 / 144) * np.exp(-br)) / r ** 3
        flsr3 = (1 - (1 + br + br ** 2 / 2 + 7 * br ** 3 / 48 + br ** 4 / 48) * np.exp(-br)) / r ** 3
        kr = me * r / hc

    fivp = -gamma - 5./6. + abs(np.log(kr)) + 6 * pi * kr / 8
    fdelta = b ** 3 * (1 + br + br ** 2 / 3) * np.exp(-br) / 16
    fnpr = b ** 3 * (15 + 15 * br + 6 * br ** 2 + br ** 3) * np.exp(-br) / 384
    
    vem[0] = alpha * hc * fcoulr
    if lpot >= 2:
        return vem

    vem[1] = -alpha * hc ** 3 * fdelta / (4 * mp ** 2)
    vem[2] = -vem[0] ** 2 / mp
    vem[3] = 2 * alpha * vem[0] * fivp / (3 * pi)
    vem[4] = alpha * hc * beta * fnpr
    vem[5] = -alpha * hc ** 3 * mup ** 2 * fdelta / (6 * mp ** 2)
    vem[6] = -alpha * hc ** 3 * mun ** 2 * fdelta / (6 * mn ** 2)
    vem[7] = -alpha * hc ** 3 * mup * mun * fdelta / (6 * mn * mp)
    vem[8] = -alpha * hc ** 3 * mup ** 2 * ftr3 / (4 * mp ** 2)
    vem[9] = -alpha * hc ** 3 * mun ** 2 * ftr3 / (4 * mn ** 2)
    vem[10] = -alpha * hc ** 3 * mup * mun * ftr3 / (4 * mp * mn)
    vem[11] = -alpha * hc ** 3 * (4 * mup - 1) * flsr3 / (2 * mp ** 2)
    vem[12] = 0
    vem[13] = -alpha * hc ** 3 * mun * flsr3 / (2 * mn * mr)

    return vem

# *id* consts **********************************************************
# subroutine for constants in av18 and sscc potentials
# ----------------------------------------------------------------------
# arguments for consts
# lpot:  input potential
# hc:    output value for hbar*c (MeV-fm)
# mpi0:    "      "    "  neutral pion mass (MeV)
# mpic:    "      "    "  charged pion mass (MeV)
# mp:      "      "    "  proton mass (MeV)
# mn:      "      "    "  neutron mass (MeV)
# alpha:   "      "    "  electromagnetic constant alpha
# mup:     "      "    "  proton magnetic moment (nm)
# mun:     "      "    "  neutron magnetic moment (nm)
# ----------------------------------------------------------------------
def consts(lpot):
    hc = 197.327053
    if lpot < 100:
        mpi0 = 134.9739
        mpic = 139.5675
        mp = 938.27231
        mn = 939.56563
    elif lpot > 100:
        mpi0 = 0.7 * hc
        mpic = 0.7 * hc
        mp = hc ** 2 / 41.47
        mn = hc ** 2 / 41.47

    alpha = 1. / 137.035989
    mup = 2.7928474
    mun = -1.9130427
    
    return hc, mpi0, mpic, mp, mn, alpha, mup, mun



# Create an array for r from 0 to 5 in steps of 0.001
r_values = np.arange(0, 5.0001, 0.0001)

# Compute V(r,1) for all r values
V_values = [V1S0(r, 1) for r in r_values]

# Plotting
plt.plot(r_values, V_values)
plt.xlabel('r')
plt.ylabel('V(r,1)')
plt.title('V as a function of r')
plt.xlim(0, 5) # Setting x-axis limits: from a to b
plt.xticks(np.arange(0, 5.01, 0.5))
plt.ylim(-120, 80) # Setting y-axis limits: from c to d
plt.yticks(np.arange(-120, 81, 20))
plt.grid(True)
plt.show()


from scipy.integrate import quad
from concurrent.futures import ProcessPoolExecutor

# Define the ranges for k and p
k_vals = np.arange(0, 50.001, 0.001)
p_vals = np.arange(0, 50.001, 0.001)


# Define the integrand function
def integrand(r, k, p):
    return V1S0(r, 1) * np.sin(k * r) * np.sin(p * r) / (k * p)

# Worker function to compute integral for a given (k,p) pair
def compute_integral(params):
    k, p = params
    val, _ = quad(integrand, 0, 20, args=(k, p))
    return val

# Create a list of (k, p) pairs
params_list = [(ki, pj) for ki in k_vals for pj in p_vals]

# Parallel integration
with ProcessPoolExecutor(max_workers=4) as executor:  # Change max_workers as needed
    results = list(executor.map(compute_integral, params_list))

# Populate the Vkp array
Vkp = np.array(results).reshape(len(k_vals), len(p_vals))