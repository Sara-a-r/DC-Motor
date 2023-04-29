"""
This file uses Euler method to solve the system of equation:
    d_t ( w ) = -b/J ( w ) + Kt/J ( i )
    d_t ( i ) = -Ke/L ( w ) - R/L ( i ) + V/L
that can be written as follows
    d_t ( u ) = A ( u ) + B
where A = [-b/J   Kt/J]  and  B = [0  ]
          [-Ke/L  -R/L]           [V/L]

Note: the inizial condition are w=0, i=0
"""

import os
import numpy as np
import matplotlib.pyplot as plt

#--------------------Setup the main directories------------------#
#Define the various directories
script_dir = os.getcwd()                         #define current dir
main_dir = os.path.dirname(script_dir)           #go up of one directory
results_dir = os.path.join(main_dir, "figure")   #define results dir

if not os.path.exists(results_dir):              #if the directory does not exist create it
    os.mkdir(results_dir)

#-----------------Integrator: Euler method------------------------#
def euler_step(u, F, dt, *params):
    return u + dt * F(u, *params)  #nd array of u at instant n+1

#--------------------------Right Hand Side------------------------#
def F(u, J, b, Kt, Ke, R, L, V):
    #define A and B
    A = np.array([[-b/J, Kt/J], [-Ke/L, -R/L]])     #matrix A of the system
    B = np.array((0, V/L))
    return A @ u + B

#----------------------------Step function------------------------#
def step_function(t, t0=0):
    return np.heaviside(t-t0, 1)    # 0 if t < t0
                                    # 1 if t >= t0

#--------------------------Temporal evolution----------------------#
def evolution(int_method, Nt_step, dt, physics_params, V, t0):
    #-----------------Initialize the problem-------------------#
    tmax = dt * Nt_step             #total time of simulation
    tt = np.arange(0, tmax, dt)     #temporal grid
    u0 = np.array((0, 0))           #initial condition
    u_t = np.copy(u0)               #create a copy to evolve it in time
    V_signal = V(tt, t0)            #signal applied to the system in time
    #----------------------------------------------------------#

    #----------------------Time evolution----------------------#
    w = []  #initialize list of w value
    i = []  #initialize list of i value
    for Vi in V_signal:
        u_t = int_method(u_t, F, dt, *physics_params, Vi) #step n+1
        w.append(u_t[0])
        i.append(u_t[1])
    return tt, np.array(w), np.array(i)


if __name__ == '__main__':
    #Parameters of the simulation
    Nt_step = 500     #temporal steps
    dt = 1e-2         #temporal step size
    #Parameters of the system
    J = 0.01    #rotor's inertia [kg m^2]
    b = 0.001   #viscous friction coefficient [N m s]
    Kt = 1      #torque constant
    Ke = 1      #electric constant
    R = 10      #resistance [Ohm]
    L = 1       #inductance [H]
    t0 = 0      #parameter of the step function [s]

    #Signal applied to the system
    V = step_function

    #Simulation
    physical_params = [J, b, Kt, Ke, R, L]
    simulation_params = [euler_step, Nt_step, dt]
    tt, w, i = evolution(*simulation_params, physical_params, V, t0)

    #--------------------------Plot results----------------------#
    plt.title('Transient response for DC motor')
    plt.xlabel('Time [s]')
    plt.ylabel('i [A]')
    plt.grid(True)
    plt.minorticks_on()
    plt.plot(tt, w, linestyle='-', linewidth=1.8, marker='')

    # save the plot in the results dir
    out_name = os.path.join(results_dir, "StepResp_w_numSim.png")
    plt.savefig(out_name)
    plt.show()

