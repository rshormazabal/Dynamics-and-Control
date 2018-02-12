import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import time

# define process model (to generate process data)
def process(y,t,n,u,Kp,taup):
    # arguments
    # y[n] = outputs
    # t = time
    # n = order of the system
    # u = input value
    # Kp = process gain
    # taup = process time constant
    
    # equations for high order system
    dydt = np.zeros(n)
    
    # calculate derivative
    dydt[0] = (-y[0] + Kp * u)/(taup/n)
    for i in range(1,n):
        dydt[i] = (-y[i] + y[i-1])/(taup/n)
    return dydt

# define first order plus dead-time approximation
def fopdt(y,t,uf,Km,taum,thetam):
    # arguments
    # y = output
    # t = time
    # uf = input linear function (for time shift)
    # Km = model gain
    # taum = model time constant
    # thetam = model dead-time
    
    # time-shift u
    try:
        if (t-thetam) <= 0:
            um = uf(0.0)
        else:
            um = uf(t-thetam)
    except:
        #print ('Error with time extrapolation: ' + str(t))
        um = 0
    # calculate derivative
    dydt = (-y + Km * um)/(taum)
    return dydt

# specify number of steps
ns = 50
# define time points
t = np.linspace(0,40,ns+1)
delta_t = t[1] - t[0]
# define input vector
u = np.zeros(ns+1)
u[5:20] = 1.0
u[20:30] = 0.1
u[30:] = 0.5
# create a linear interpolation of the u data vs time    
uf = interp1d(t,u)

# use this function or replace yp with real process data
def sim_process_data():
    # higher order process
    n = 10   # order
    Kp = 3.0   # gain
    taup = 5.0 # time constant
    
    # storage for predictions or data
    yp = np.zeros(ns+1) # process
    for i in range(1,ns+1):
        if i ==1:
            yp0 = np.zeros(n)
        ts = [delta_t*(i-1),delta_t*i]
        y = odeint(process, yp0, ts, args = (n,u[i],Kp,taup))
        yp0 = y[-1]
        yp[i] = y[1][n-1]
    return yp

yp = sim_process_data()

# simulate FOPDT model with x = [Km,taum,thetam]
def sim_model(x):
    # arguments
    Km = x[0]
    taum = x[1]
    thetam = x[2]
    # storage for model values
    ym = np.zeros(ns+1)
    # initial condition
    ym[0] = 0
    
    # loop through time steps
    for i in range(1,ns+1):
        ts = [delta_t*(i-1),delta_t*i]
        y1 = odeint(fopdt,ym[i-1],ts,args=(uf,Km,taum,thetam))
        ym[i] = y1[-1]
    return ym

# define objective for optimization
def objective(x):
    # simulate model
    ym = sim_model(x)
    #calculate objective
    obj = 0.0
    for i in range(len(ym)):
        obj += (ym[i]-yp[i])**2
    #return result
    return obj

# initial guesses
x0 = np.zeros(3)
x0[0] = 3 # Km
x0[1] = 5 # taum
x0[2] = 1 # thetam

# show initial objective
print('Initial SSE objective: ' + str(objective(x0)))

start = time.clock()

# optimize solution
solution = minimize(objective,x0)
x = solution.x

print('Optimization process time = ' + str(time.clock() - start) + ' seconds')

# printing optimization results
print('optimized SSE objective: ' + str(objective(x)))
print('Optimized FOPDT parameters: ')
print('Km = ' + str(x[0]))
print('taum = ' + str(x[1]))
print('thetam = ' + str(x[2]))


# calculate model with updated parameters
ym1 = sim_model(x0)
ym2 = sim_model(x)

#plot results

plt.figure()
plt.subplot(2,1,1)
plt.plot(t,ym1,'b-',linewidth=2,label='Initial guess')
plt.plot(t,ym2,'r--',linewidth=3,label='Optimized FOPDT')
plt.plot(t,yp,'kx-',linewidth=2,label='Process Data')
plt.ylabel('Output')
plt.legend(loc='best')
plt.subplot(2,1,2)
plt.plot(t,u,'bx-',linewidth=2)
plt.plot(t,uf(t),'r--',linewidth=3)
plt.legend(['Measured','Interpolated'],loc='best')
plt.ylabel('Input Data')
plt.show()


