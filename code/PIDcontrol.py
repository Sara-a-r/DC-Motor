"""
This file uses Euler method to solve the system of equation:
    d_t ( w ) = -b/J ( w ) + Kt/J ( i )
    d_t ( i ) = -Ke/L ( w ) - R/L ( i ) + V/L
    d_t ( theta ) = ( w )
that can be written as follows
    d_t ( u ) = A ( u ) + B
where A = [-b/J   Kt/J  0]  and  B = [0  ]
          [-Ke/L  -R/L  0]           [V/L]
          [1        0   0]           [0  ]
And it uses PID control to control the system variables.

Note: the initial condition are w=0, i=0, theta=0
"""

import os
import numpy as np
import matplotlib.pyplot as plt

#--------------------Setup the main directories------------------#
#Define the various directories
script_dir = os.getcwd()                         #define current dir
main_dir = os.path.dirname(script_dir)           #go up of one directory
results_dir = os.path.join(main_dir, "figure")   #define results dir
data_dir = os.path.join(main_dir, "data")        #define data dir

if not os.path.exists(results_dir):              #if the directory does not exist create it
    os.mkdir(results_dir)

#-----------------Integrator: Euler method------------------------#
def euler_step(u, F, dt, *params):
    return u + dt * F(u, *params)  #nd array of u at instant n+1

#--------------------------Right Hand Side------------------------#
def F(u, J, b, Kt, Ke, R, L, V):
    #define A and B
    A = np.array([[-b/J, Kt/J, 0], [-Ke/L, -R/L, 0], [1, 0, 0]])     #matrix A of the system
    B = np.array((0, V/L, 0))
    return A @ u + B

#--------------------------Temporal evolution----------------------#
def evolution(int_method, Nt_step, dt, physics_params, w_ref, kp, ki, kd):
    #-----------------Initialize the problem-------------------#
    tmax = dt * Nt_step             #total time of simulation
    tt = np.arange(0, tmax, dt)     #temporal grid
    u0 = np.array((0, 0, 0))        #initial condition
    u_t = np.copy(u0)               #create a copy to evolve it in time
    #----------------------------------------------------------#

    #----------------------Time evolution----------------------#
    w = []         #initialize list of w value
    i = []         #initialize list of i value
    theta = []     #initialize list of theta value
    err_t = []     #list for memorizing the value of the error
    V = 0
    I = 0
    j = 0
    for t in tt:
        u_t = int_method(u_t, F, dt, *physics_params, V) #step n+1
        w.append(u_t[0])
        i.append(u_t[1])
        theta.append(u_t[2])
        err = w_ref - u_t[0]        #evaluate the error
        err_t.append(err)
        delta_err = err - err_t[j-1]
        P = kp * err                    #calculate P term
        I = I + ki * ( err * dt )       #calculate the I term
        D = kd * ( delta_err / dt )     #calculate the D term
        V = P + I + D                   #calculate PID term
        j = j+1

    return tt, np.array(w), np.array(i), np.array(theta)


if __name__ == '__main__':
    #Parameters of the simulation
    Nt_step = 5e3     #temporal steps
    dt = 1e-3         #temporal step size
    #Parameters of the system
    J = 1.13e-2   #rotor's inertia [N m s^2/ rad]
    b = 0.028     #viscous friction coefficient [N m s/rad]
    Kt = 0.067    #torque constant [N m/amp]
    Ke = 0.067    #electric constant [V s/rad]
    R = 0.45      #resistance [Ohm]
    L = 1e-1      #inductance [H]

    #Parameters of control
    w_ref = 1
    kp = 3
    ki = 15
    kd = 0.1

    #Simulation
    physical_params = [J, b, Kt, Ke, R, L]
    simulation_params = [euler_step, Nt_step, dt]
    control_params = [w_ref, kp, ki, kd]
    tt, w, i, theta = evolution(*simulation_params, physical_params, *control_params)

    #save data
    #np.savetxt('../data/PIDcontrol_kd0.1.txt', w, fmt='%.5f')

    #--------------------------Plot results----------------------#
    plt.title('P-control for DC motor')
    plt.xlabel('Time [s]')
    plt.ylabel('$\omega$ [rad/s]')
    plt.grid(True)
    plt.minorticks_on()
    plt.plot(tt, w, linestyle='-', linewidth=1.8, marker='')

    # save the plot in the results dir
    out_name = os.path.join(results_dir, "testPcontrol.png")
    #plt.savefig(out_name)
    plt.show()

