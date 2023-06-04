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

#--------------------------Temporal evolution----------------------#
def evolution(evol_method, Nt_step, dt, physical_params, ref, kp, ki, kd):
    #-----------------Initialize the problem-------------------#
    tmax = dt * Nt_step             #total time of simulation
    tt = np.arange(0, tmax, dt)     #temporal grid
    y0 = np.array((0, 0, 0))           #initial condition
    y_t = np.copy(y0)               #create a copy to evolve it in time
    #----------------------------------------------------------#

    #----------------------Time evolution----------------------#
    w = []      #initialize list of w value
    i = []      #initialize list of i value
    theta = []  #initialize list of theta value

    err_t = []  # list for memorizing the value of the error
    V = 0
    I = 0
    j = 0

    A, B = matrix(*physical_params)     #evaluate matrix for AR model

    for t in tt:
        y_t = evol_method(y_t, A, B, V) #step n+1
        w.append(y_t[0])
        i.append(y_t[1])
        theta.append(y_t[2])

        err = ref - y_t[0]  # evaluate the error
        err_t.append(err)
        delta_err = err - err_t[j - 1]
        P = kp * err  # calculate P term
        I = I + ki * (err * dt)  # calculate the I term
        D = kd * (delta_err / dt)  # calculate the D term
        V = P + I + D  # calculate PID term
        j = j + 1

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
    K = 0.5       #spring constant [N/m]
    t0 = 0      #parameter of the step function [s]

    #Parameters of control
    ref = 0.5  #theta

    # Simulation
    physical_params = [J, b, Kt, Ke, R, L, K, dt]
    simulation_params = [AR_model, Nt_step, dt]

    control_params_PID = [ref, 600, 500, 5]     #w_ref, kp, ki, kd

    tt_PID, w_PID, i_PID, theta_PID = evolution(*simulation_params, physical_params, *control_params_PID)

    # --------------------------Plot results----------------------#
    plt.rc('font', size=12)
    plt.figure(figsize=(10, 6))
    plt.title('PID control for DC motor')
    plt.xlabel('Time [s]')
    plt.ylabel('$\Theta$ [rad]')
    plt.grid(True)
    plt.minorticks_on()

    plt.axhline(y=ref, linestyle=':', color='red', linewidth=1.3, label='$\Theta_{ref}$ = %.1f rad' % ref)
    plt.plot(tt_PID, w_PID, linestyle='-', linewidth=1.3, marker='', color='black',
             label='PID control\nk$_p$ = %d, k$_i$ = %d, k$_d$ = %d' % (control_params_PID[1], control_params_PID[2], control_params_PID[3]))
    plt.legend()

# save the plot in the results dir
out_name = os.path.join(results_dir, "PID_k.png")
plt.savefig(out_name)
plt.show()

