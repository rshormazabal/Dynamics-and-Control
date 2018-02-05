import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.pyplot import style

style.use('fivethirtyeight')

# car variables
m = 500.0     #kg
Cd = 0.24   # Drag coeficient
rho = 1.225 #kg/m3 - Air density
A = 5.0       #m2 - Vehicle cross sectional area
Fp = 30.0     #N/%pesal - Thurst parameter

# define model
def vehicle(v,t,u,load):
    dvdt = (1.0/(m+load)) * (Fp * u  - 0.5 * rho * A * Cd * v**2)
    return dvdt
    
# initial conditions
v0 = 0.0

# time vector
# final time
tf = 300

# number of time points
nsteps = tf * 1 + 1

# time points
t = np.linspace(0,tf,nsteps)

# simulation step test operation
step = np.zeros(nsteps) # u = valve % open 

# cargo load
load = 200.0  # kg

# store solution
vs = np.zeros(nsteps)

# Set point
sp_store = np.zeros(nsteps) # to track SP
sp = 25 # initial SP
sp_store[0] = sp

# PI controller parameters
ubias = 0.0
Kc =  1.0/1.2 * 5 # Proportional gain
tauI = 30.0 # tau integral
tauD = 0.9 # tau derivative
sum_int = 0.0 # integral sum
es = np.zeros(nsteps) # error
ies = np.zeros(nsteps) # integral error

# solve ODE
for i in range(nsteps-1):
    # span for the next time step
    tspan = [t[i],t[i+1]]
    delta_t = t[i+1] - t[i]
    
    # schedule change in SP
    if i == 50:
        sp = 0
    if i == 100:
        sp = 25
    if i == 150:
        sp = 30
    if i == 200:
        sp = 10
    sp_store[i+1] = sp
    
    # controller
    error = sp - v0
    es[i+1] = error
    sum_int = sum_int + error * delta_t
    u = ubias + Kc * error + Kc/tauI * sum_int + Kc * tauD * (es[i+1] - es[i])/delta_t
    
    # clip inputs to -50% to 100%
    if u >= 100.0:
        u = 100.0
        sum_int = sum_int - error * delta_t # anti reset windup
    if u <= -50.0:
        u = -50.0 
        sum_int = sum_int - error * delta_t # anti reset windup
        
    ies[i+1] = sum_int
    step[i+1] = u
    
    # solve for next step
    v = odeint(vehicle,v0,tspan, args=(u,load)) #args must be tuple, PUT COMA!
    
    # next initial condition
    v0 = v[-1]
    
    # store solution for plotting
    vs[i+1] = v0
    

# plot results
plt.figure()
plt.subplot(2,2,1)
plt.plot(t,vs,'b-', linewidth = 3)
plt.plot(t,sp_store,'k--',linewidth = 2)
plt.ylabel('Velocity (m/s)')
plt.legend(['Velocity','Set point'], loc = 'best')
plt.subplot(2,2,2)
plt.plot(t,step,'r--', linewidth = 3)
plt.ylabel('Gas pedal')
plt.legend(['Gas pedal (%)'], loc = 'best')
plt.subplot(2,2,3)
plt.plot(t,es,'b-', linewidth = 3)
plt.ylabel('Error (SP-PV)')
plt.xlabel('Time (sec)')
plt.subplot(2,2,4)
plt.plot(t,ies,'k--', linewidth = 3)
plt.ylabel('Integral of error')
plt.xlabel('Time (sec)')




