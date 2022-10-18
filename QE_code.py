# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:42:26 2022

@author: mattk
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

'''
read in all data and seperate
'''
data = pd.read_csv('QE_data.csv')
light = data['light av']
current = data['current average']
light_error = data['light std dev']
current_error = data['std deviation']


#seperate by colour

g_light = light[0:14]*(10**-3)
p_light = light[15:27]*(10**-3)
y_light = light[28:37]*(10**-3)
b_light = light[38:53]*(10**-3)

g_light_err = light_error[0:14]*(10**-3)
p_light_err = light_error[15:27]*(10**-3)
y_light_err = light_error[28:37]*(10**-3)
b_light_err = light_error[38:53]*(10**-3)

g_curr = current[0:14]*10**-8
p_curr = current[15:27]*10**-8
y_curr = current[28:37]*10**-8
b_curr = current[38:53]*10**-8

g_curr_err = current_error[0:14]*10**-8
p_curr_err = current_error[15:27]*10**-8
y_curr_err = current_error[28:37]*10**-8
b_curr_err = current_error[38:53]*10**-8


#straight line
def straight(m,x,c):
    return m*x+c

#fit all data
pg,pgcov = curve_fit(straight, g_light, g_curr, sigma  = g_curr_err)
pp,ppcov = curve_fit(straight, p_light, p_curr, sigma  = p_curr_err)
py,pycov = curve_fit(straight, y_light, y_curr, sigma  = y_curr_err)
pb,pbcov = curve_fit(straight, b_light, b_curr, sigma  = b_curr_err)
xg = np.linspace(0,0.002,100)
xp = np.linspace(0,0.00006,100)
xy = np.linspace(0,10*10**-4,100)
xb = np.linspace(0,0.00045,100)
'''
plot all data

'''


fig, axs = plt.subplots(2, 2)
axs[0, 0].errorbar(g_light, g_curr,g_curr_err,g_light_err,'o',color = 'green',capsize = 4)
axs[0, 0].plot(xg,xg*pg[0]+pg[1])
axs[0, 0].set_title("green")
axs[0, 0].grid()
axs[1, 0].errorbar(p_light,p_curr,p_curr_err,p_light_err,'o',color = 'purple',capsize = 4)
axs[1, 0].plot(xp,xp*pp[0]+pp[1])
axs[1, 0].set_title("purple")
axs[1, 0].grid()
axs[0, 1].errorbar(y_light, y_curr,y_curr_err,y_light_err,'o',color = 'yellow',capsize = 4)
axs[0, 1].plot(xy,xy*py[0]+py[1])
axs[0, 1].set_title("yellow")
axs[0, 1].grid()
axs[1, 1].errorbar(b_light,b_curr,b_curr_err,b_light_err,'o',color = 'blue',capsize = 4)
axs[1, 1].plot(xb,xb*pb[0]+pb[1])
axs[1, 1].set_title("blue")
axs[1, 1].grid()
fig.tight_layout()
plt.show()
'''
print(pg[0],pgcov,'green')
print(pp[0],ppcov,'purple')
print(py[0],pycov,'yellow')
print(pb[0],pbcov,'blue')
'''
#calculating QE
mg, mp, my, mb = pg[0], pp[0], py[0], pb[0]


def cal_QE(wavelength, gradient):
    h = 6.63 * 10 ** -34
    e = 1.6 * 10 ** -19
    c = 3 * 10 ** 8
    return (h * c* gradient) / (e * wavelength)

qg, qp, qy, qb = cal_QE(546*10**-9, mg), cal_QE(405*10**-9, mp), cal_QE(578*10**-9,my), cal_QE(436*10**-9, mb)

plt.plot(546, qg*100,'o',color =  'green')
plt.plot(405, qp*100,'o',color =  'purple')
plt.plot(578, qy*100,'o',color =  'yellow')
plt.plot(436, qb*100,'o',color =  'blue')
plt.grid()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Quantum Efficiency (%)')








  


