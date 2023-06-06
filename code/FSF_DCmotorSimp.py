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


# --------------------------Right Hand Side------------------------#
def F(u, J, b, Kt, Ke, R, L, r, N):
    # define A and B
    alpha11 = -b / J
    alpha12 = Kt / J
    alpha21 = - Ke / L
    alpha22 = - R / L
    beta = 1 / L

    # FSF parameters
    k2 = (10 + alpha22 + alpha11) / beta
    k1 = (26 - alpha11 * alpha22 + alpha11 * beta * k2 + alpha12 * alpha21) / (alpha12 * beta)

    # system matrices
    A = np.array([[alpha11, alpha12], [alpha21, alpha22]])  # matrix A of the system
    B = np.array((0, beta))
    K = np.array((k1, k2))

    return (A - np.outer(B, K)) @ u + B * N * r


# --------------------------Temporal evolution----------------------#
def evolution(int_method, Nt_step, dt, physics_params):
    # -----------------Initialize the problem-------------------#
    tmax = dt * Nt_step  # total time of simulation
    tt = np.arange(0, tmax, dt)  # temporal grid
    u0 = np.array((0, 0))  # initial condition
    u_t = np.copy(u0)  # create a copy to evolve it in time
    # ----------------------------------------------------------#

    # ----------------------Time evolution----------------------#
    w = []  # initialize list of w value
    i = []  # initialize list of i value
    t = 0
    for t in tt:
        u_t = int_method(u_t, F, dt, *physics_params)  # step n+1
        w.append(u_t[0])
        i.append(u_t[1])
        t = t + 1
    return tt, np.array(w), np.array(i)


if __name__ == '__main__':
    # Parameters of the simulation
    Nt_step = 3e2  # temporal steps
    dt = 1e-2  # temporal step size
    # Parameters of the system
    J = 0.01  # rotor's inertia [kg m^2]
    b = 0.1  # viscous friction coefficient [N m s]
    Kt = 0.01  # torque constant
    Ke = 0.01  # electric constant
    R = 1  # resistance [Ohm]
    L = 0.5  # inductance [H]

    r = 1  # w ref
    N = 13  # factor for scaling the input

    # Simulation
    physical_params = [J, b, Kt, Ke, R, L, r, N]
    simulation_params = [euler_step, Nt_step, dt]
    tt, w, i = evolution(*simulation_params, physical_params)

    # --------------------------Plot results----------------------#
    plt.title('FSF control')
    plt.xlabel('Time [s]')
    plt.ylabel('$\omega$ [rad/s]')
    plt.grid(True)
    plt.minorticks_on()

    plt.axhline(y=r, linestyle=':', color='red', linewidth=1.3, label='$\omega_{ref}$ = %d rad/s' % r)
    plt.plot(tt, w, linestyle='-', linewidth=1.4, marker='')
    plt.legend()

    # save the plot in the results dir
    out_name = os.path.join(results_dir, "FSF_DCmotorSimp.png")
    plt.savefig(out_name)
    plt.show()