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

#--------------------------AR model------------------------#
def AR_model(y, A, B, u):
    return A @ y + B * u  #nd array of y at instant n+1

#--------------------------Right Hand Side------------------------#
def matrix(J, b, Kt, Ke, R, L, K, dt):
    alpha11 = 1 - b * dt / J
    alpha12 = Kt * dt / J
    alpha13 = - K * dt / J
    alpha21 = - Ke * dt / L
    alpha22 = 1 - R * dt / L
    alpha23 = 0
    alpha31 = dt
    alpha32 = 0
    alpha33 = 1
    beta = 1 * dt / L

    #define A and B
    A = np.array([[alpha11, alpha12, alpha13], [alpha21, alpha22, alpha23],
                  [alpha31, alpha32, alpha33]])
    B = np.array((0, beta, 0))
    return A, B

#----------------------------Step function------------------------#
def step_function(t, t0=0):
    return np.heaviside(t-t0, 1)    # 0 if t < t0
                                    # 1 if t >= t0

#--------------------------Temporal evolution----------------------#
def evolution(evol_method, Nt_step, dt, physical_params, V, t0):
    #-----------------Initialize the problem-------------------#
    tmax = dt * Nt_step             #total time of simulation
    tt = np.arange(0, tmax, dt)     #temporal grid
    y0 = np.array((0, 0, 0))           #initial condition
    y_t = np.copy(y0)               #create a copy to evolve it in time
    V_signal = V(tt, t0)            #signal applied to the system in time
    #----------------------------------------------------------#

    #----------------------Time evolution----------------------#
    w = []      #initialize list of w value
    i = []      #initialize list of i value
    theta = []  #initialize list of theta value

    A, B = matrix(*physical_params)
    for Vi in V_signal:
        y_t = evol_method(y_t, A, B, Vi) #step n+1
        w.append(y_t[0])
        i.append(y_t[1])
        theta.append(y_t[2])
    return tt, np.array(w), np.array(i), np.array(theta)


if __name__ == '__main__':
    #Parameters of the simulation
    Nt_step = 2e3     #temporal steps
    dt = 1e-3         #temporal step size
    #Parameters of the system
    J = 0.01    #rotor's inertia [kg m^2]
    b = 0.001   #viscous friction coefficient [N m s]
    Kt = 1      #torque constant
    Ke = 1      #electric constant
    R = 10      #resistance [Ohm]
    L = 1       #inductance [H]
    #K = 1       #spring constant [N/m]
    t0 = 0      #parameter of the step function [s]

    #Signal applied to the system
    V = step_function

    K_list = [0.01, 0.1, 0.5, 1, 2]

    for K in K_list:
        # Simulation
        physical_params = [J, b, Kt, Ke, R, L, K, dt]
        simulation_params = [AR_model, Nt_step, dt]
        tt, w, i, theta = evolution(*simulation_params, physical_params, V, t0)

        # --------------------------Plot results----------------------#
        plt.title('Step response for DC motor (AR model)')
        plt.xlabel('Time [s]')
        plt.ylabel('$\omega$ [rad/s]')
        plt.grid(True)
        plt.minorticks_on()
        plt.plot(tt, w, linestyle='-', linewidth=1.2, marker='', label=r'K = %0.2f' % K)
        plt.legend()

    #save the plot in the results dir
    out_name = os.path.join(results_dir, "StepResp_theta_w.png")
    plt.savefig(out_name)
    plt.show()

