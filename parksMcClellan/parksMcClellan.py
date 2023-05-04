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
## find the extreme points
#  @param E pointer to error function
#  @param n number of extreme points
#  @param d calculated error
#  @param xtol x tolerance
#  @param ytol y tolerance
#  @return array of extreme points
#
#-------------------------------------------------------------------------------
def findExtrema(E, n, d, wtol = 1e-6, ytol = 1e-5):
    x   = np.cos(np.linspace(0, np.pi, num = int(np.pi/wtol)))
    err = np.absolute(E(x))
    k   = (err > ytol)
    k   = np.insert(k, 0, False)
    k   = np.append(k, False)
    k   = np.diff(k)
    up  = np.argwhere(k)[::2,0]
    dwn = np.argwhere(k)[1::2,0]
    ind = np.array([], dtype = np.int64)
    assert len(up) == len(dwn), "Something went wrong"
    for j in range(0, len(up)):
        jmax = np.argmax(err[up[j]:dwn[j]])
        ind  = np.append(ind, jmax + up[j])
    i = np.flip(np.argsort(err[ind]))
    extrema = x[ind[i[0:(n+2)]]]
    extrema = np.flip(extrema[np.argsort(extrema)])
    if len(extrema) < n + 2:
        plt.plot(np.arccos(x), 
                 np.log10(err))
        plt.plot(np.arccos(extrema), 
                 np.log10(np.absolute(E(extrema))), 
                 marker = "o")
        plt.xlabel("cos(w)")
        plt.ylabel("log10(abs(error))")
        plt.title("Couldn't find all local min/max")
        plt.show()
        assert False, "Something went wrong"
    return extrema

#-------------------------------------------------------------------------------
## parksMcClellan algorithm
#  @param n order of the filter
#  @param maxiter maximum iteration
#  @param step x axis resolution for calculation of the extrema
#  @return error and filter coefficients
#
#-------------------------------------------------------------------------------
def parksMcClellan(H, n, maxiter = 100, eacc = 0.0001, wtol = 1e-6, ytol = 1e-5):
    assert n%2 == 0, "n must be even" 
    n = int(n/2)
    extrema = np.cos(np.linspace(0, np.pi, num=(n+2), dtype = np.float64))
    pm = np.array([(-1.0)**(i%2)  for i in range(1, len(extrema))])
    d = delta(extrema, H)    
    old_d = 1000
    lagrange_int = lagrange(extrema[:-1], H(extrema[:-1]) + d*pm)
    iterations = 0
    while (old_d/d > (1 + eacc/100.0) or old_d/d < (1 - eacc/100.0)) and \
           iterations < maxiter:
        extrema = findExtrema(lambda x: (H(x) - lagrange_int(x)), n, d, wtol, ytol)
        old_d = d
        d = delta(extrema, H)    
        lagrange_int = lagrange(extrema[:-1], H(extrema[:-1]) + d*pm)
        iterations = iterations + 1
    x = np.linspace(1, -1, num=10000)
    plt.plot(x, H(x)-lagrange_int(x))
    plt.xlabel('Cos(w)')
    plt.ylabel('Error')
    plt.show()
    bk = Polynomial(lagrange_int.coef[::-1]).coef   
    tk = np.polynomial.chebyshev.poly2cheb(bk)
    hk = np.append(np.flip(tk[1:]/2.0), tk[0])
    hk = np.append(hk, tk[1:]/2.0)
    return iterations, d, hk

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
    n = 20
    H = lambda x: Hbp(np.cos(wstop1), np.cos(wpass1), \
                      np.cos(wpass2), np.cos(wstop2), x)
    iterations, d, hk = parksMcClellan(H, n, ytol = 1e-4)
    print("Filter coefficients: " + str(hk))
    print("Error: " + str(abs(d)))
    print("Iterations: " + str(abs(iterations)))
    w, h = signal.freqz(hk)
    fig = plt.figure()
    plt.title('Band pass digital filter frequency response')
    ax1 = fig.add_subplot(111)
    plt.plot(w/np.pi, abs(h), 'b')
    plt.ylabel('Amplitude [mag]', color='b')
    plt.xlabel('Frequency ratio [rad/(sample*pi)]')
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    plt.plot(w/np.pi, angles/np.pi*180, 'g')
    plt.ylabel('Angle (degree)', color='g')
    plt.grid()
    plt.axis('tight')
    plt.show()

    #---------------------------------------------------------------------------
    # Low pass filter design
    #---------------------------------------------------------------------------
    n = 50
    wpass2 = 0.8*np.pi
    wstop2 = 0.85*np.pi
    H = lambda x: Hlp(np.cos(wpass2), np.cos(wstop2), x)

    iterations, d, hk = parksMcClellan(H, n)
    print("Filter coefficients: " + str(hk))
    print("Error: " + str(abs(d)))
    print("Iterations: " + str(abs(iterations)))
    w, h = signal.freqz(hk)
    fig = plt.figure()
    plt.title('Low pass digital filter frequency response')
    ax1 = fig.add_subplot(111)
    plt.plot(w/np.pi, abs(h), 'b')
    plt.ylabel('Amplitude [mag]', color='b')
    plt.xlabel('Frequency ratio [rad/(sample*pi)]')
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    plt.plot(w/np.pi, angles/np.pi*180, 'g')
    plt.ylabel('Angle (degree)', color='g')
    plt.grid()
    plt.axis('tight')
    plt.show()

    #---------------------------------------------------------------------------
    # High pass filter design
    #---------------------------------------------------------------------------
    n = 30
    wstop1 = 0.5*np.pi
    wpass1 = 0.6*np.pi
    H = lambda x: Hhp(np.cos(wstop1), np.cos(wpass1), x)

    iterations, d, hk = parksMcClellan(H, n)
    print("Filter coefficients: " + str(hk))
    print("Error: " + str(abs(d)))
    print("Iterations: " + str(abs(iterations)))
    w, h = signal.freqz(hk)
    fig = plt.figure()
    plt.title('High pass digital filter frequency response')
    ax1 = fig.add_subplot(111)
    plt.plot(w/np.pi, abs(h), 'b')
    plt.ylabel('Amplitude [mag]', color='b')
    plt.xlabel('Frequency ratio [rad/(sample*pi)]')
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    plt.plot(w/np.pi, angles/np.pi*180, 'g')
    plt.ylabel('Angle (degree)', color='g')
    plt.grid()
    plt.axis('tight')
    plt.show()

    #---------------------------------------------------------------------------
    # Fancy filter
    #---------------------------------------------------------------------------
    n = 40
    H = lambda x: Hexp(x)
    iterations, d, hk = parksMcClellan(H, n)
    print("Filter coefficients: " + str(hk))
    print("Error: " + str(abs(d)))
    print("Iterations: " + str(abs(iterations)))
    w, h = signal.freqz(hk)
    fig = plt.figure()
    plt.title('High pass digital filter frequency response')
    ax1 = fig.add_subplot(111)
    plt.plot(w/np.pi, abs(h), 'b')
    plt.ylabel('Amplitude [mag]', color='b')
    plt.xlabel('Frequency ratio [rad/(sample*pi)]')
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    plt.plot(w/np.pi, angles/np.pi*180, 'g')
    plt.ylabel('Angle (degree)', color='g')
    plt.grid()
    plt.axis('tight')
    plt.show()
