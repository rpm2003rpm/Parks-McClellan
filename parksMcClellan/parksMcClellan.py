## @package parksMcClellan
#  Simple implementation of the parksMcClellan algorithm. Convergence is painful 
#  for filters of order 50 or higher.
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
#  @param x x points
#  @param error calculated error
#  @param order of the polynomial
#  @return array of extreme points
#
#-------------------------------------------------------------------------------
def findExtrema(x, error, n):
    deriv = np.ediff1d(error)
    index = np.where(np.diff(np.sign(deriv)))[0]
    if deriv[0]*error[0] < 0:
        index = np.insert(index, 0, 0)
    if deriv[-1]*error[-1] > 0:
        index = np.append(index, len(error) - 1)
    y = np.absolute(error[index])
    i = np.flip(np.argsort(y))
    extrema = x[index[i[0:(n+2)]]]
    extrema = np.flip(extrema[np.argsort(extrema)])
    extrema = np.unique(extrema)
    assert len(extrema) == n + 2, "Something went wrong"
    return extrema

#-------------------------------------------------------------------------------
## parksMcClellan algorithm
#  @param n order of the filter
#  @param step x axis resolution for calculation of the extrema
#  @return error and filter coefficients
#
#-------------------------------------------------------------------------------
def parksMcClellan(H, n, step = 1e-5):
    assert n%2 == 0, "n must be even" 
    n = int(n/2)
    x = np.linspace(1, -1, num = int(2.0/step), dtype = np.float64)
    extrema = np.cos(np.linspace(0, np.pi, num=(n+2), dtype = np.float64))
    pm = np.array([(-1.0)**(i%2)  for i in range(1, len(extrema))])
    d = delta(extrema, H)    
    old_d = 1000
    lagrange_int = lagrange(extrema[:-1], H(extrema[:-1]) + d*pm)
    iterations = 0
    while old_d/d > 1.0000001 or old_d/d < 0.9999999:
        extrema = findExtrema(x, H(x) - lagrange_int(x), n)
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
## Main
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    n = 30
    wstop1 = 0.3*np.pi
    wpass1 = 0.4*np.pi
    wpass2 = 0.6*np.pi
    wstop2 = 0.7*np.pi

    #---------------------------------------------------------------------------
    # Band pass design
    #---------------------------------------------------------------------------
    H = lambda x: Hbp(np.cos(wstop1), np.cos(wpass1), \
                      np.cos(wpass2), np.cos(wstop2), x)

    iterations, d, hk = parksMcClellan(H, n)
    print("Filter coefficients: " + str(hk))
    print("Error: " + str(abs(d)))
    print("Iterations: " + str(abs(iterations)))
    w, h = signal.freqz(hk)
    fig = plt.figure()
    plt.title('Digital filter frequency response')
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
    wpass2 = 0.8*np.pi
    wstop2 = 0.85*np.pi
    H = lambda x: Hlp(np.cos(wpass2), np.cos(wstop2), x)

    iterations, d, hk = parksMcClellan(H, n)
    print("Filter coefficients: " + str(hk))
    print("Error: " + str(abs(d)))
    print("Iterations: " + str(abs(iterations)))
    w, h = signal.freqz(hk)
    fig = plt.figure()
    plt.title('Digital filter frequency response')
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
    wstop1 = 0.5*np.pi
    wpass1 = 0.6*np.pi
    H = lambda x: Hhp(np.cos(wstop1), np.cos(wpass1), x)

    iterations, d, hk = parksMcClellan(H, n)
    print("Filter coefficients: " + str(hk))
    print("Error: " + str(abs(d)))
    print("Iterations: " + str(abs(iterations)))
    w, h = signal.freqz(hk)
    fig = plt.figure()
    plt.title('Digital filter frequency response')
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
