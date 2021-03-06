import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import style

style.use('seaborn-deep')

# defining termocouple
def termocouple(x,t,Tg,Tf):
    Tt = x[0]
    TtLin = x[1]
    
    # defining system caracteristics
    h =2800.0 # W/(m2 K) - heat transfer coefficient
    rho_t = 20.0 # g/cm3 - termocuple density
    cp_t= 0.1 # J /(g K) -  heat capacity
    d_t = 0.001 # cm - diameter termocouple bead
    A_t = 4.0 * np.pi * ((d_t/2.0)/100.0)**2 # m2 - area termocouple bead
    V_t = 4.0/3.0 * np.pi * (d_t/2)**3 # cm3 - volume termocouple bead

    sigma = 5.67e-8 # W/(m2 K4)- the Stefan-Boltzmann constant
    eps =0.8 # emissivity
    
    # acc = inlet - outlet
    # acc = m * Cp * dT/dt = rho * V * Cp * dT/dt
    # inlet - outlet from 2 heat transfer pathways
    # q(convection) = h * A * (Tg-Tt)
    # q(radiation) = A * esp * sigma * (Tf^4 - Tt^4)
    q_conv = h * A_t * (Tg - Tt)
    q_rad = A_t * eps * sigma * (Tf**4 - Tt**4)
    dTtdt = (q_conv + q_rad)/(rho_t * V_t * cp_t )
    
    Tt0 = 1500.0
    Tg0 = 1500.0
    Tf0 = 1500.0
    
    alpha = - h * A_t - 4 * A_t * eps * sigma * Tt**3
    beta = h * A_t
    gamma =  4 * A_t * eps * sigma * Tt**3
    denom = rho_t * V_t * cp_t 
    dTt_Lindt = alpha/denom * (TtLin - Tt0) + beta/denom * (Tg - Tg0) + gamma/denom * (Tf - Tf0)
    return [dTtdt, dTt_Lindt]

# starting termocouple temperature
Tt_0 = [1500.0,1500.0]

# time vector
tf = 0.10
nsteps = tf * 10000 + 1
t = np.linspace(0,tf,nsteps)

# defining flame temperature
Tf = 1500 * np.ones(len(t))
Tf[501:] = 1520.0

# defining gas temperature
Tg_amp = 75 # C - Amplitude of the flame
Tg_freq = 100 # Hz - Frequency of the flame
Tg = Tf + Tg_amp * np.sin(Tg_freq * (2*np.pi) * t) # K

# saving profile
Ts = np.ones(len(t)) * Tf
TLins = np.ones(len(t)) * Tf

# solve differential equation

for i in range(len(t)-1):
    ts = [t[i],t[i+1]]
    T = odeint(termocouple,Tt_0,ts,args=(Tg[i],Tf[i]))
    Tt_0 = T[-1]
    Ts[i+1] = Tt_0[0]
    TLins[i+1] = Tt_0[1]
    
# plot the results
plt.figure()
plt.plot(t,Tg,'b--',linewidth=3,label='Gas Temperature')
plt.plot(t,Tf,'g:',linewidth=3,label='Flame Temperature')
plt.plot(t,Ts,'r-',linewidth=3,label='Thermocouple')
plt.plot(t,TLins,'k--',linewidth=3,label='Linear')
plt.ylabel('Temperature (K)')
plt.legend(loc='best')
plt.xlim([0,t[-1]])
plt.xlabel('Time (sec)')

plt.show()










