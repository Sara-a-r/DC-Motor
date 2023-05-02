import numpy as np
import pylab as pyl
import matplotlib.pyplot as plt

kp = [1, 3, 5, 15]        #proportional gain

#load data
w1 = pyl.loadtxt(f'../data/Pcontrol_kp{kp[0]}.txt', unpack=True)
w2 = pyl.loadtxt(f'../data/Pcontrol_kp{kp[1]}.txt', unpack=True)
w3 = pyl.loadtxt(f'../data/Pcontrol_kp{kp[2]}.txt', unpack=True)
w4 = pyl.loadtxt(f'../data/Pcontrol_kp{kp[3]}.txt', unpack=True)


Nt_step = 5e3       # temporal steps
dt = 1e-3           # temporal step size
tmax = dt * Nt_step # total time of simulation
tt = np.arange(0, tmax, dt)  # temporal grid

# let's see data: scatter plot
plt.rc('font',size=10)
plt.figure(figsize=(12,6))
plt.title("Proportional control varing $k_p$")
plt.xlabel("time [s]")
plt.ylabel('$\omega$ [rad/s]')
plt.grid(visible=True, color='grey', linestyle='-', alpha=0.3)
plt.minorticks_on()

w_ref = 1
w = [w1, w2, w3, w4]

plt.axhline(y=w_ref, linestyle=':', color='red', label= '$\omega_{ref}$ = %d rad/s' % w_ref)
i = 0
for i in range(len(w)):
    plt.plot(tt, w[i], linestyle='-', linewidth=1.5, marker='', label=r'$k_p$ = %d' % kp[i])
    plt.legend(loc='lower right', fancybox=True)
    i = i + 1

#save the plot
plt.savefig(f"../figure/Pcontrol_kp.png")
plt.show()
