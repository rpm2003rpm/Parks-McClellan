## @package parksMcClellan
#  Simple implementation of the parksMcClellan algorithm. Convergence is painful 
#  for in a few conditions. Not quite sure why.
# 
#  @author  Rodrigo Pedroso Mendes
#  @version V1.0
#  @date    02/05/23 18:31:45
#
#  #LICENSE# 
#    
#  Copyright (c) 2023 Rodrigo Pedroso Mendes
#
#  Permission is hereby granted, free of charge, to any  person   obtaining  a 
#  copy of this software and associated  documentation files (the "Software"), 
#  to deal in the Software without restriction, including  without  limitation 
#  the rights to use, copy, modify,  merge,  publish,  distribute, sublicense, 
#  and/or sell copies of the Software, and  to  permit  persons  to  whom  the 
#  Software is furnished to do so, subject to the following conditions:        
#   
#  The above copyright notice and this permission notice shall be included  in 
#  all copies or substantial portions of the Software.                         
#   
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,  EXPRESS OR 
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE  WARRANTIES  OF  MERCHANTABILITY, 
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
#  AUTHORS OR COPYRIGHT HOLDERS BE  LIABLE FOR ANY  CLAIM,  DAMAGES  OR  OTHER 
#  LIABILITY, WHETHER IN AN ACTION OF  CONTRACT, TORT  OR  OTHERWISE,  ARISING 
#  FROM, OUT OF OR IN CONNECTION  WITH  THE  SOFTWARE  OR  THE  USE  OR  OTHER  
#  DEALINGS IN THE SOFTWARE. 
#    
################################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
from scipy import signal

#-------------------------------------------------------------------------------
## TF of low pass filter
#-------------------------------------------------------------------------------
def Hlp(xpass, xstop, x):
    assert xstop >= -1 or xstop <= 1,    "xstop is out of range"
    assert xpass >= -1 or xpass <= 1,    "xpass is out of range"
    assert ((x >= -1) | (x <= 1)).all(), "x is out of range"
    y= np.zeros_like(x)
    y[x >= xpass] = 1.0
    mask = (x < xpass) & (x > xstop) 
    y[mask] = (np.arccos(x[mask]) - np.arccos(xstop))/ \
              (np.arccos(xpass)   - np.arccos(xstop))
    return y

#-------------------------------------------------------------------------------
## TF of high pass filter
#-------------------------------------------------------------------------------
def Hhp(xstop, xpass, x):
    return 1.0 - Hlp(xstop, xpass, x)

#-------------------------------------------------------------------------------
## TF of a band pass filter
#-------------------------------------------------------------------------------
def Hbp(xstop1, xpass1, xpass2, xstop2, x):
    return Hlp(xpass2, xstop2, x)*Hhp(xstop1, xpass1, x)

#-------------------------------------------------------------------------------
## TF of a fancy filter
#-------------------------------------------------------------------------------
def Hexp(x):
    return np.exp(-2*np.arccos(x)/np.pi)

#-------------------------------------------------------------------------------
## Gamma
#-------------------------------------------------------------------------------
def gk(k, extrema):
    return 1.0/(np.prod(extrema[k] - extrema[0:k])*\
                np.prod(extrema[k] - extrema[(k+1):len(extrema)]))
    
#-------------------------------------------------------------------------------
## Delta 
#-------------------------------------------------------------------------------
def delta(extrema, H):
    gks = np.array([gk(k, extrema) for k in range(0, len(extrema))])
    pm  = np.array([(-1.0)**(i%2)  for i in range(0, len(extrema))])
    return np.sum(gks*H(extrema))/np.sum(gks*pm)
        
#-------------------------------------------------------------------------------
## Find the extremal points. More experimentation is needed. It doesn't seems 
#  to work for filters of high order
#  @param E pointer to error function
#  @param m number of extreme points
#  @param d calculated error
#  @param xtol x tolerance
#  @param ytol y tolerance
#  @return array of extreme points
#
#-------------------------------------------------------------------------------
def findExtrema(E, m, d, wtol = 1e-5, ytol = 1e-7, debug = True):
    #Build an array of x points and calculate both the error and the differences
    #Dont touch
    x   = np.cos(np.linspace(0, np.pi, num = int(np.pi/wtol)))
    err = E(x)
    dif = np.diff(err)
    
    #Eliminate differences lower than the tolerance
    maskth = np.absolute(dif) > ytol
    ith    = np.argwhere(maskth)[:,0]
    difth  = dif[maskth]
    xth = x[ith]
    eth = err[ith]

    #Find the concavities and the edges
    sgn = np.zeros_like(difth)
    sgn[difth >= 0] = 1.0
    con = np.diff(sgn) 
    edg = np.argwhere(con != 0)[:,0]
    con = con[edg]
    ext = xth[edg]

    #Decide if the upper and lower bounds need to be added
    #Skip if upper bound is already an extremal point
    if (ext < xth[1]).all():
        if (eth[0] - eth[edg[0]])*con[0] > ytol:
            edg = np.insert(edg, 0, 0)
            con = np.insert(con, 0, -con[0])      
    #Skip if lower bound is already an extremal point
    if (ext > xth[-1]).all():
        if (err[-1] - eth[edg[-1]])*con[-1] > ytol:
            edg = np.append(edg, len(xth) - 1)
            con = np.append(con, -con[-1])      
    ext = xth[edg]

    #If an even number of points can be removed, remove the lower diff 
    #between min/max
    rm = int((len(ext) - m)/2)
    if rm > 0:
        diffe = np.diff(eth[edg])
        for i in range(0, rm):
            irm = np.argsort(np.absolute(diffe))[0]
            lrm = np.array([irm, irm + 1], dtype = np.int32)
            if irm != (len(diffe) - 1) and irm != 0:
                diffe[irm - 1] = diffe[irm - 1] + diffe[irm + 1]
            if irm == (len(diffe) - 1):
                diffe = np.delete(diffe, lrm - 1)
            else:
                diffe = np.delete(diffe, lrm)
            edg   = np.delete(edg, lrm)
            con   = np.delete(con, lrm)
            ext   = np.delete(ext, lrm)

    #If we still need to remove one point, remove one of the lower or
    #upper bounds
    if len(ext) > m:
        if abs(eth[edg[0]] - eth[edg[1]]) > abs(eth[edg[-1]] - eth[edg[-2]]):
            ext = ext[:-1]  
            edg = edg[:-1]  
            con = con[:-1]  
        else:
            ext = ext[1:]  
            edg = edg[1:]  
            con = con[1:]  
            
    #Double check alternation of concavities. Should hold by construction.
    pm = np.array([(-1.0)**(i%2)  for i in range(0, len(con))])
    assert (np.diff(pm*con) == 0).all(), "Ops.... " 

    #Debug and error
    if debug or len(ext) != m:
        up  = edg[con > 0]
        dwn = edg[con < 0] 
        plt.plot(np.arccos(x), err)
        plt.scatter(np.arccos(xth[up]),  eth[up])
        plt.scatter(np.arccos(xth[dwn]), eth[dwn])
        plt.xlabel("cos(w)")
        plt.ylabel("log10(abs(error))")
        if len(ext) == m:    
            plt.title("Debug mode. Extremal points found.")
        else:
            plt.title("Couldn't find all the necessary extremal points")
        plt.show()
    assert len(ext) == m, "Something went wrong"
    return ext

#-------------------------------------------------------------------------------
## parksMcClellan algorithm
#  @param n order of the filter
#  @param maxiter maximum iteration
#  @param step x axis resolution for calculation of the extrema
#  @return error and filter coefficients
#
#-------------------------------------------------------------------------------
def parksMcClellan(H, n, maxiter = 100, \
                   eacc = 0.0001, wtol = 1e-4, ytol = 1e-8,
                   debug = True):
    assert n%2 == 0, "n must be even" 
    n = int(n/2)
    extrema = np.cos(np.linspace(0, np.pi, num=(n+4), dtype = np.float64)[1:-1])
    pm = np.array([(-1.0)**(i%2)  for i in range(1, len(extrema))])
    d = delta(extrema, H)    
    old_d = 1000
    lagrange_int = lagrange(extrema[:-1], H(extrema[:-1]) + d*pm)
    iterations = 0
    while (old_d/d > (1 + eacc/100.0) or old_d/d < (1 - eacc/100.0)) and \
           iterations < maxiter:
        extrema = findExtrema(lambda x: (H(x) - lagrange_int(x)), \
                              n + 2, d, wtol, ytol, debug)
        old_d = d
        d = delta(extrema, H)    
        lagrange_int = lagrange(extrema[:-1], H(extrema[:-1]) + d*pm)
        iterations = iterations + 1
    bk = Polynomial(lagrange_int.coef[::-1]).coef   
    tk = np.polynomial.chebyshev.poly2cheb(bk)
    hk = np.append(np.flip(tk[1:]/2.0), tk[0])
    hk = np.append(hk, tk[1:]/2.0)
    return iterations, d, hk


#-------------------------------------------------------------------------------
# Filter plot
#-------------------------------------------------------------------------------
def filterPlot(hk, H, title):
    w, h = signal.freqz(hk)
    fig = plt.figure()
    plt.title(title)
    ax1 = fig.add_subplot(111)
    plt.plot(w/np.pi, abs(h), 'b')
    plt.plot(w/np.pi, H(np.cos(w)), 'gray')
    plt.ylabel('Amplitude [mag]', color='b')
    plt.xlabel('Frequency ratio [rad/(sample*pi)]')
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    plt.plot(w/np.pi, angles/np.pi*180, 'g')
    plt.ylabel('Angle (degree)', color='g')
    plt.grid()
    plt.axis('tight')
    plt.show()

#-------------------------------------------------------------------------------
## Main
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    wstop1 = 0.3*np.pi
    wpass1 = 0.4*np.pi
    wpass2 = 0.6*np.pi
    wstop2 = 0.7*np.pi

    #---------------------------------------------------------------------------
    # Band pass design
    #---------------------------------------------------------------------------
    n = 18
    H = lambda x: Hbp(np.cos(wstop1), np.cos(wpass1), \
                      np.cos(wpass2), np.cos(wstop2), x)
    iterations, d, hk = parksMcClellan(H, n, debug = False)
    print("Filter coefficients: " + str(hk))
    print("Error: " + str(abs(d)))
    print("Iterations: " + str(abs(iterations)))
    filterPlot(hk, H, "Band pass frequency response")

    #---------------------------------------------------------------------------
    # Low pass filter design
    #---------------------------------------------------------------------------
    n = 50
    wpass2 = 0.8*np.pi
    wstop2 = 0.85*np.pi
    H = lambda x: Hlp(np.cos(wpass2), np.cos(wstop2), x)
    iterations, d, hk = parksMcClellan(H, n, debug = False)
    print("Filter coefficients: " + str(hk))
    print("Error: " + str(abs(d)))
    print("Iterations: " + str(abs(iterations)))
    filterPlot(hk, H, "Low  pass frequency response")

    #---------------------------------------------------------------------------
    # High pass filter design
    #---------------------------------------------------------------------------
    n = 26
    wstop1 = 0.5*np.pi
    wpass1 = 0.6*np.pi
    H = lambda x: Hhp(np.cos(wstop1), np.cos(wpass1), x)

    iterations, d, hk = parksMcClellan(H, n, debug = False)
    print("Filter coefficients: " + str(hk))
    print("Error: " + str(abs(d)))
    print("Iterations: " + str(abs(iterations)))
    filterPlot(hk, H, "High pass frequency response")

    #---------------------------------------------------------------------------
    # Fancy filter
    #---------------------------------------------------------------------------
    n = 40
    H = lambda x: Hexp(x)
    iterations, d, hk = parksMcClellan(H, n, debug = False)
    print("Filter coefficients: " + str(hk))
    print("Error: " + str(abs(d)))
    print("Iterations: " + str(abs(iterations)))
    filterPlot(hk, H, "Fancy filter  frequency response")
