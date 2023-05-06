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
## Constant weight function
#  @param x input vector
#  @return output of the same size as the input vector fill-up with ones
#
#-------------------------------------------------------------------------------
def Wconst(x):
    assert isinstance(x, np.ndarray), "x must be a numpy array" 
    assert ((x >= -1) | (x <= 1)).all(), "x is out of range"
    return np.ones_like(x)   

#-------------------------------------------------------------------------------
## Weight function of a low pass filter
#  @param xpass pass frequency as a function of cos(wpass)
#  @param xstop stop frequency as a function of cos(stop)
#  @param x input vector
#  @return np array representing the weight
#
#-------------------------------------------------------------------------------
def Wlp(xpass, xstop, x):
    assert xstop >= -1 or xstop <= 1, "xstop is out of range"
    assert xpass >= -1 or xpass <= 1, "xpass is out of range"
    assert isinstance(x, np.ndarray), "x must be a numpy array" 
    assert ((x >= -1) | (x <= 1)).all(), "x is out of range"
    y= np.ones_like(x)
    mask = (x < xpass) & (x > xstop)
    #avg = (np.arccos(xpass) + np.arccos(xstop))/2.0
    #y[mask] = 0.02 + 0.98*np.absolute((np.arccos(x[mask]) - avg)/ \
    #                                  (np.arccos(xpass)   - avg))
    y[mask] = 0.1
    return y
    
#-------------------------------------------------------------------------------
## Transfer function of a low pass filter
#  @param xpass pass frequency as a function of cos(wpass)
#  @param xstop stop frequency as a function of cos(stop)
#  @param x input vector
#  @return np array representing the TF
#
#-------------------------------------------------------------------------------
def Hlp(xpass, xstop, x):
    assert xstop >= -1 or xstop <= 1, "xstop is out of range"
    assert xpass >= -1 or xpass <= 1, "xpass is out of range"
    assert isinstance(x, np.ndarray), "x must be a numpy array" 
    assert ((x >= -1) | (x <= 1)).all(), "x is out of range"
    y= np.zeros_like(x)
    y[x >= xpass] = 1.0
    mask = (x < xpass) & (x > xstop) 
    y[mask] = (np.arccos(x[mask]) - np.arccos(xstop))/ \
              (np.arccos(xpass)   - np.arccos(xstop))
    return y

#-------------------------------------------------------------------------------
## Transfer function of a high pass filter
#  @param xstop stop frequency as a function of cos(stop)
#  @param xpass pass frequency as a function of cos(wpass)
#  @param x input vector
#  @return np array representing the TF
#
#-------------------------------------------------------------------------------
def Hhp(xstop, xpass, x):
    return 1.0 - Hlp(xstop, xpass, x)

#-------------------------------------------------------------------------------
## Transfer function of a band pass filter
#  @param xstop1 1st stop frequency as a function of cos(stop1)
#  @param xpass1 1st pass frequency as a function of cos(wpass1)
#  @param xpass2 2st pass frequency as a function of cos(wpass2)
#  @param xstop2 2st stop frequency as a function of cos(stop2)
#  @param x input vector
#  @return np array representing the TF
#
#-------------------------------------------------------------------------------
def Hbp(xstop1, xpass1, xpass2, xstop2, x):
    return Hlp(xpass2, xstop2, x)*Hhp(xstop1, xpass1, x)

#-------------------------------------------------------------------------------
## TF of a filter for teesting
#  @param x input vector
#  @return np array representing the TF
#
#-------------------------------------------------------------------------------
def Hexp(x):
    assert isinstance(x, np.ndarray), "x must be a numpy array" 
    assert ((x >= -1) | (x <= 1)).all(), "x is out of range"
    return np.exp(-2*np.arccos(x)/np.pi)

#-------------------------------------------------------------------------------
## Gamma
#  @param k idex
#  @param extremal frequencies
#  @return gamma k
#
#-------------------------------------------------------------------------------
def gk(k, extremal):
    assert isinstance(extremal, np.ndarray), "extremal must be a numpy array" 
    assert isinstance(k, int), "k must be int" 
    assert k >= 0 and k < len(extremal), "k is outside limits" 
    return 1.0/(np.prod(extremal[k] - extremal[0:k])*\
                np.prod(extremal[k] - extremal[(k+1):len(extremal)]))
    
#-------------------------------------------------------------------------------
## Delta 
#  @param extremal frequencies
#  @param H transfer function 
#  @param W weight function 
#  @return real number representing the delta
#
#-------------------------------------------------------------------------------
def delta(extremal, H, W):
    assert isinstance(extremal, np.ndarray), "extremal must be a numpy array" 
    assert callable(H), "H must be callable"
    assert callable(W), "W must be callable"
    gks = np.array([gk(k, extremal) for k in range(0, len(extremal))])
    pm  = np.array([(-1.0)**(i%2)  for i in range(0, len(extremal))])
    return np.sum(gks*H(extremal))/np.sum(gks/W(extremal)*pm)
        
#-------------------------------------------------------------------------------
## Find the extremal points. More experimentation is needed. It doesn't seems 
#  to work for filters of high order
#  @param E pointer to weighted error function
#  @param m number of extreme points
#  @param d calculated error
#  @param xtol x tolerance
#  @param ytol y tolerance
#  @param debug enable or disable the debug mode
#  @return array of extremal points
#
#-------------------------------------------------------------------------------
def findExtremal(E, m, d, wtol = 1e-5, ytol = 1e-7, debug = True):
    assert callable(E), "E must be callable"
    assert isinstance(m, int), "number of extreme points must be integer" 
    assert isinstance(d, float), "delta must be float" 
    assert isinstance(ytol, float), "ytol must be float" 
    assert isinstance(wtol, float), "wtol must be float" 
    assert isinstance(debug, bool), "debug must be bool"
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
    #Skip if upper bound is already an extremall point
    if (ext < xth[1]).all():
        if (eth[0] - eth[edg[0]])*con[0] > ytol:
            edg = np.insert(edg, 0, 0)
            con = np.insert(con, 0, -con[0])      
    #Skip if lower bound is already an extremall point
    if (ext > xth[-1]).all():
        if (err[-1] - eth[edg[-1]])*con[-1] > ytol:
            edg = np.append(edg, len(xth) - 1)
            con = np.append(con, -con[-1])      
    ext = xth[edg]

    #Debug and error
    if debug:
        up  = edg[con > 0]
        dwn = edg[con < 0] 
        plt.plot(np.arccos(x), err)
        plt.scatter(np.arccos(xth[up]),  eth[up])
        plt.scatter(np.arccos(xth[dwn]), eth[dwn])
        plt.xlabel("cos(w)")
        plt.ylabel("log10(abs(Weighted error))")
        plt.title("Candidate to extremal points.")
        plt.show()

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
        plt.ylabel("log10(abs(Weighted error))")
        if len(ext) == m:    
            plt.title("Debug mode. Extremall points found.")
        else:
            plt.title("Couldn't find all the necessary extremall points")
        plt.show()
    assert len(ext) == m, "Something went wrong"
    return ext

#-------------------------------------------------------------------------------
## parksMcClellan algorithm
#  @param H transfer function 
#  @param W weight function 
#  @param n order of the filter
#  @param maxiter maximum numbe of iterations
#  @param eacc the algorithm will stop when the error changes between
#         iterations is less than eacc%
#  @param xtol x tolerance
#  @param ytol y tolerance
#  @param debug enable or disable the debug mode
#  @param iniExt inital extremal points. It will be linear if equal to noe
#  @return error and filter coefficients
#
#-------------------------------------------------------------------------------
def parksMcClellan(H, W, n, maxiter = 100, \
                   eacc = 0.0001, wtol = 1e-4, ytol = 1e-8,
                   debug = True,
                   iniExt = None):
    assert callable(H), "H must be callable"
    assert callable(W), "W must be callable"
    assert isinstance(n, int), "filter order must be integer" 
    assert isinstance(maxiter, int), "maximum iteration must be integer" 
    assert isinstance(eacc, float), "eacc must be float" 
    assert isinstance(ytol, float), "ytol must be float" 
    assert isinstance(wtol, float), "wtol must be float" 
    assert isinstance(debug, bool), "debug must be bool"
    assert n%2 == 0, "n must be even" 
    m = int(n/2)
    if iniExt == None:
        extremal = np.cos(np.linspace(0, np.pi, num = (m + 4)))[1:-1] 
    else:
        extremal = iniExt
        assert isinstance(iniExt, np.ndarray), "iniExt must be an ndarray" 
    pm = np.array([(-1.0)**(i%2)  for i in range(1, len(extremal))])
    d = delta(extremal, H, W)    
    old_d = 1000
    lagrange_int = lagrange(extremal[:-1], \
                            H(extremal[:-1]) + pm*d/W(extremal[:-1]))
    iterations = 0
    while (old_d/d > (1 + eacc/100.0) or old_d/d < (1 - eacc/100.0)) and \
           iterations < maxiter:
        extremal = findExtremal(lambda x: (H(x) - lagrange_int(x))*W(x), \
                               m + 2, d, wtol, ytol, debug)
        old_d = d
        d = delta(extremal, H, W)    
        lagrange_int = lagrange(extremal[:-1], \
                                H(extremal[:-1]) + pm*d/W(extremal[:-1]))
        iterations = iterations + 1
    bk = Polynomial(lagrange_int.coef[::-1]).coef   
    tk = np.polynomial.chebyshev.poly2cheb(bk)
    hk = np.append(np.flip(tk[1:]/2.0), tk[0])
    hk = np.append(hk, tk[1:]/2.0)
    return iterations, d, hk


#-------------------------------------------------------------------------------
## Filter plot
#  @param H transfer function 
#  @param hk filter impulse response
#  @param title title
#
#-------------------------------------------------------------------------------
def filterPlot(hk, H, title):
    assert isinstance(hk, np.ndarray), "hk must be an ndarray" 
    assert callable(H), "H must be callable"
    assert isinstance(title, str), "title must be a string" 
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
    iterations, d, hk = parksMcClellan(H, Wconst, n, debug = False)
    print("Filter coefficients: " + str(hk))
    print("Error: " + str(abs(d)))
    print("Iterations: " + str(abs(iterations)))
    filterPlot(hk, H, "Band pass frequency response")

    #---------------------------------------------------------------------------
    # Low pass filter design
    #---------------------------------------------------------------------------
    n = 8
    wpass2 = 0.3*np.pi
    wstop2 = 0.8*np.pi
    H = lambda x: Hlp(np.cos(wpass2), np.cos(wstop2), x)
    W = lambda x: Wlp(np.cos(wpass2), np.cos(wstop2), x)
    iterations, d, hk = parksMcClellan(H, W, n, debug = False)
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
    iterations, d, hk = parksMcClellan(H, Wconst, n, debug = False)
    print("Filter coefficients: " + str(hk))
    print("Error: " + str(abs(d)))
    print("Iterations: " + str(abs(iterations)))
    filterPlot(hk, H, "High pass frequency response")

    #---------------------------------------------------------------------------
    # Fancy filter
    #---------------------------------------------------------------------------
    n = 40
    H = lambda x: Hexp(x)
    iterations, d, hk = parksMcClellan(H, Wconst, n, debug = False)
    print("Filter coefficients: " + str(hk))
    print("Error: " + str(abs(d)))
    print("Iterations: " + str(abs(iterations)))
    filterPlot(hk, H, "Fancy filter  frequency response")
