import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum, diff_to_matrix, sub_symbols
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.lqr.lqr_controller import LQRController

def simulate_params(mpar_con, Q, R, x0):
    robot = "pendubot"

    mpar_con.set_motor_inertia(0.0)
    mpar_con.set_damping([0.0, 0.0])
    mpar_con.set_cfric([0.0, 0.0])

    if robot == "pendubot":
        torque_limit = [0.15, 0.0]
        active_act = 0
    elif robot == "acrobot":
        torque_limit = [0.0, 0.15]
        active_act = 1
    mpar_con.set_torque_limit(torque_limit) # type: ignore
    print(mpar_con)

    dt = 0.002
    tf = 10
    integrator = "runge_kutta"

    plant = SymbolicDoublePendulum(model_pars=mpar_con)
    sim = Simulator(plant=plant)


    goal = [np.pi, 0, 0, 0]
    controller = LQRController(model_pars=mpar_con)
    controller.set_goal(goal)
    controller.set_cost_matrices(Q=Q, R=R)
    controller.set_parameters(failure_value=0, cost_to_go_cut=10000)

    controller.init()

    T, X, U, anim = sim.simulate_and_animate(
        t0=0.0,
        x0=x0,
        tf=tf,
        dt=dt,
        controller=controller,
        integrator=integrator
    )

    return anim
