import os
import numpy as np
import matplotlib.pyplot as plt

# --------------------Setup the main directories------------------#
# Define the various directorieszz
script_dir = os.getcwd()  # define current dir
main_dir = os.path.dirname(script_dir)  # go up of one directory
results_dir = os.path.join(main_dir, "figure")  # define results dir

if not os.path.exists(results_dir):  # if the directory does not exist create it
    os.mkdir(results_dir)


#------------------------FSF AR model------------------------#
def FSF_AR_model(y, A, B, K, N, r):
    return (A - np.outer(B, K)) @ y + B * N * r   #nd array of y at instant n+1


# -----------------Matrices of the system--------------------#
def matrix(dt, J, b, Kt, Ke, R, L, K, p, wn, epsilon):
    # define A and B matrices of the system
    a11 = 1 - dt * b / J
    a12 = dt * Kt / J
    a13 = - dt * K / J
    a21 = - dt * Ke / L
    a22 = 1 - dt * R / L
    a23 = 0
    a31 = dt
    a32 = 0
    a33 = 1
    beta = dt / L

    A = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
    B = np.array((0, beta, 0))

    # FSF parameters
    gamma1 = -2*np.cos(wn*np.sqrt(1-epsilon**2)*dt)*np.exp(-wn*epsilon*dt)-np.exp(p*dt)
    gamma2 = np.exp(-2*wn*epsilon*dt)-2*np.cos(wn*np.sqrt(1-epsilon**2)*dt)*np.exp(-wn*epsilon*dt+p*dt)
    gamma3 = np.exp(-2*wn*epsilon*dt+p*dt)

    k2 = (gamma1 + a22 + a11 + a33) / beta
    k1 = (gamma2 - a11 * a22 + a11 * beta * k2 - a22 * a33 + beta * k2 * a33 - a11 * a33 + a13 * a31 + a21 * a12) / (a12 * beta)
    k3 = (gamma3 + a11 * a22 * a33 - a11 * a33* beta * k2 - a13 * a22 * a31 + a13 * beta * k2 * a31 - a33 * a12 * a21 + a12 * beta * k1 * a33) / (a12 * beta * a31)

    K = np.array((k1, k2, k3))

    return A, B, K


# --------------------------Temporal evolution----------------------#
def evolution(evol_method, Nt_step, dt, physical_params, control_params, pole_params):
    #-----------------Initialize the problem-------------------#
    tmax = dt * Nt_step             #total time of simulation
    tt = np.arange(0, tmax, dt)     #temporal grid
    y0 = np.array((0, 0, 0))           #initial condition
    y_t = np.copy(y0)               #create a copy to evolve it in time

    #----------------------Time evolution----------------------#
    w = []      #initialize list of w value
    i = []      #initialize list of i value
    theta = []  #initialize list of theta value

    A, B, K = matrix(dt, *physical_params, *pole_params)

    print(K)
    print(B)
    print(A)
    print(np.outer(B, K))

    for t in tt:
        y_t = evol_method(y_t, A, B, K, *control_params) #step n+1
        w.append(y_t[0])
        i.append(y_t[1])
        theta.append(y_t[2])
    return tt, np.array(w), np.array(i), np.array(theta)


if __name__ == '__main__':
    # Parameters of the simulation
    Nt_step = 5e2  # temporal steps
    dt = 1e-2  # temporal step size
    # Parameters of the system
    J = 0.01  # rotor's inertia [kg m^2]
    b = 0.001  # viscous friction coefficient [N m s]
    Kt = 1  # torque constant
    Ke = 1  # electric constant
    R = 10  # resistance [Ohm]
    L = 1  # inductance [H]
    K = 0.5  # spring constant [N/m]

    # control parameters
    r = 0.5  # theta ref
    N = 4050  # factor for scaling the input

    # parameters for desired poles
    p = -8                                 #real pole
    epsilon = 0.6                        #damping coefficient
    wn = 7*np.pi/(10*dt)                       #natural frequency
    sigma = wn * epsilon                  #real part of pole
    wd = wn * np.sqrt( 1 - epsilon**2)    #imaginary part of pole

    # Simulation
    pole_params = [p, wn, epsilon]
    physical_params = [J, b, Kt, Ke, R, L, K]
    control_params = [N, r]
    simulation_params = [FSF_AR_model, Nt_step, dt]
    tt, w, i, theta = evolution(*simulation_params, physical_params, control_params, pole_params)

    # --------------------------Plot results----------------------#
    plt.rc('font', size=13)
    plt.figure(figsize=(10, 6))
    plt.title(r'$\xi$=%.1f, $\omega_n=%.1f$, (poles s-plane: p$_{1,2}$=-%.1f$\pm$%.1f,  p$_3$=%d)'
              % (epsilon, wn, sigma, wd, p))
    plt.xlabel('Time [s]')
    plt.ylabel(r'$\Theta$ [rad]')
    plt.grid(True)
    plt.minorticks_on()

    plt.axhline(y=r, linestyle=':', color='red', linewidth=1.3, label=r'$\Theta_{ref}$ = %.1f rad' % r)
    plt.plot(tt, theta, linestyle='-', linewidth=1.4, marker='', label='FSF control')
    plt.legend()

    #save the plot in the results dir
    #out_name = os.path.join(results_dir, "FSF_DCmotorK.png")
    #plt.savefig(out_name)
    plt.show()