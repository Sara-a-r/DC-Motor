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
def matrix(dt, J, b, Kt, Ke, R, L, K, p, sigma, wd):
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
    k2 = ((-2*sigma-p) + a22 + a11 + a33) / beta
    k1 = ((2*sigma*p+sigma**2+wd**2) - a11 * a22 + a11 * beta * k2 - a22 * a33 + beta * k2 * a33 - a11 * a33 + a13 * a31 + a21 * a12) / (a12 * beta)
    k3 = ((-p*sigma**2-p*wd**2) + a11 * a22 * a33 - a11 * a33* beta * k2 - a13 * a22 * a31 + a13 * beta * k2 * a31 - a33 * a12 * a21 + a12 * beta * k1 * a33) / (a12 * beta * a31)

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
    Nt_step = 10e3  # temporal steps
    dt = 1e-3  # temporal step size
    # Parameters of the system
    J = 0.01  # rotor's inertia [kg m^2]
    b = 0.001  # viscous friction coefficient [N m s]
    Kt = 1  # torque constant
    Ke = 1  # electric constant
    R = 10  # resistance [Ohm]
    L = 1  # inductance [H]
    K = 0.5  # spring constant [N/m]

    # control parameters
    r = 1  # theta ref
    N = 10  # factor for scaling the input

    # parameters for desired poles
    p = 1        # real pole
    sigma = 0.1    # real part of pole
    wd = 0.1    # imaginary part of pole
    epsilon = 0.7

    # Simulation
    pole_params = [p, sigma, wd]
    physical_params = [J, b, Kt, Ke, R, L, K]
    control_params = [N, r]
    simulation_params = [FSF_AR_model, Nt_step, dt]
    tt, w, i, theta = evolution(*simulation_params, physical_params, control_params, pole_params)

    # --------------------------Plot results----------------------#
    plt.rc('font', size=12)
    plt.figure(figsize=(10, 6))
    plt.title(r'$\xi$=%.1f, (poles : p$_{1,2}$=-%d$\pm$%d,  p$_3$=%d)'
              % (epsilon, sigma, wd, p))
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