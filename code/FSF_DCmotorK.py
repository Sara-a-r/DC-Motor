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


# -----------------Integrator: Euler method------------------------#
def euler_step(u, F, dt, *params):
    return u + dt * F(u, *params)  # nd array of u at instant n+1


# -----------------Right Hand Side: FSF control--------------------#
def F(u, J, b, Kt, Ke, R, L, K, r, N):
    # define A and B matrices of the system
    a11 = -b / J
    a12 = Kt / J
    a13 = - K / J
    a21 = - Ke / L
    a22 = - R / L
    a23 = 0
    a31 = 1
    a32 = 0
    a33 = 0
    beta = 1 / L

    A = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
    B = np.array((0, beta, 0))

    # FSF parameters
    k2 = (26 + a22 + a11) / beta
    k1 = (315 + a11 * beta * k2 + a13 * a31 + a21 * a12) / (a12 * beta)
    k3 = (1352 - a13 * a22 * a31 + a13 * beta * k2 * a31) / (a12 * beta * a31)

    K = np.array((k1, k2, k3))

    return (A - np.outer(B, K)) @ u + B * N * r


# --------------------------Temporal evolution----------------------#
def evolution(int_method, Nt_step, dt, physics_params, control_params):
    # -----------------Initialize the problem-------------------#
    tmax = dt * Nt_step  # total time of simulation
    tt = np.arange(0, tmax, dt)  # temporal grid
    u0 = np.array((0, 0, 0))  # initial condition
    u_t = np.copy(u0)  # create a copy to evolve it in time
    # ----------------------------------------------------------#

    # ----------------------Time evolution----------------------#
    w = []  # initialize list of w value
    i = []  # initialize list of i value
    theta = []  # initialize list of theta value
    t = 0
    for t in tt:
        u_t = int_method(u_t, F, dt, *physics_params, *control_params)  # step n+1
        w.append(u_t[0])
        i.append(u_t[1])
        theta.append(u_t[2])
        t = t + 1
    return tt, np.array(w), np.array(i), np.array(theta)


if __name__ == '__main__':
    # Parameters of the simulation
    Nt_step = 3e2  # temporal steps
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
    r = 1  # theta ref
    N = 13.5  # factor for scaling the input

    # Simulation
    physical_params = [J, b, Kt, Ke, R, L, K]
    control_params = [r, N]
    simulation_params = [euler_step, Nt_step, dt]
    tt, w, i, theta = evolution(*simulation_params, physical_params, control_params)

    # --------------------------Plot results----------------------#
    plt.rc('font', size=12)
    plt.figure(figsize=(10, 6))
    plt.title('FSF control DC motor and spring')
    plt.xlabel('Time [s]')
    plt.ylabel('$\Theta$ [rad]')
    plt.grid(True)
    plt.minorticks_on()

    plt.axhline(y=r, linestyle=':', color='red', linewidth=1.3, label='$\Theta_{ref}$ = %.1f rad' % r)
    plt.plot(tt, theta, linestyle='-', linewidth=1.4, marker='', label='FSF control')
    plt.legend()

    #save the plot in the results dir
    out_name = os.path.join(results_dir, "FSF_DCmotorK.png")
    plt.savefig(out_name)
    plt.show()