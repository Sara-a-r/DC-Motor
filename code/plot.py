import numpy as np
import pylab as pyl
import matplotlib.pyplot as plt

kd = [0.1, 0.3, 1, 2]        #proportional gain

#load data
w1 = pyl.loadtxt(f'../data/PIDcontrol_kd{kd[0]}.txt', unpack=True)
w2 = pyl.loadtxt(f'../data/PIDcontrol_kd{kd[1]}.txt', unpack=True)
w3 = pyl.loadtxt(f'../data/PIDcontrol_kd{kd[2]}.txt', unpack=True)
w4 = pyl.loadtxt(f'../data/PIDcontrol_kd{kd[3]}.txt', unpack=True)


Nt_step = 5e3       # temporal steps
dt = 1e-3           # temporal step size
tmax = dt * Nt_step # total time of simulation
tt = np.arange(0, tmax, dt)  # temporal grid

# let's see data: scatter plot
plt.rc('font',size=12)
plt.figure(figsize=(10,6))
plt.title("PID control varing $k_d$")
plt.xlabel("time [s]")
plt.ylabel('$\omega$ [rad/s]')
plt.grid(visible=True, color='grey', linestyle='-', alpha=0.3)
plt.minorticks_on()

w_ref = 1
w = [w1, w2, w3, w4]

plt.axhline(y=w_ref, linestyle=':', color='red', label= '$\omega_{ref}$ = %d rad/s' % w_ref)
i = 0
for i in range(len(w)):
    plt.plot(tt, w[i], linestyle='-', linewidth=1.5, marker='', label=r'$k_d$ = %0.1f' % kd[i])
    plt.legend(loc='lower right', fancybox=True)
    i = i + 1

#save the plot
plt.savefig(f"../figure/PIDcontrol_kd.png")
plt.show()
