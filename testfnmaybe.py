import numpy as np
import matplotlib.pyplot as plt
from neurodiffeq import diff, ode
from neurodiffeq.conditions import IVP
from neurodiffeq.solvers import Solver1D
from scipy.integrate import odeint
from neurodiffeq.monitors import Monitor1D
from neurodiffeq.networks import FCNN # fully-connect neural network
from neurodiffeq.networks import SinActv # sin activation
from neurodiffeq.generators import Generator1D
from torch.optim import SGD




a=0.7
gamma=0.8
epsilon=0.2
Iext=0.75




fn= lambda v, w, t: [diff(v, t) - v * (v - a) * (1 - v) + w - Iext,
diff(w, t) - epsilon * (v - gamma * w),]
       


# specify the initial conditions
init_vals_pc = [
IVP(t_0=0.0, u_0=0.0),
IVP(t_0=0.0, u_0=0.0)
]

nets_lv = [
FCNN(n_input_units=1, n_output_units=1, hidden_units=(80, 40, 80), actv=SinActv),
FCNN(n_input_units=1, n_output_units=1, hidden_units=(80, 40, 80), actv=SinActv)

]
    

# Set up the FitzHugh-Nagumo ODE

monitor = Monitor1D(t_min=0.0, t_max=100, check_every=100)
# ... and turn it into a Callback instance
monitor_callback = monitor.to_callback()

train_gen = Generator1D(size=64, t_min=0.0, t_max=100, method='uniform')
valid_gen = Generator1D(size=128, t_min=0.0, t_max=100, method='equally-spaced')


# Define the solver
solver = Solver1D(
    ode_system=fn,
    conditions=init_vals_pc,
    t_min=0,
    t_max=100,
    nets=nets_lv,
    train_generator=train_gen,
    valid_generator=valid_gen,
    


    
)

internals = solver.get_internals()


# Solve the ODE
solver.fit(max_epochs=20000, callbacks=[monitor_callback])


# Plot the results
t = np.linspace(0, 100, 100)
solution_lv = solver.get_solution()
v_net, w_net = solution_lv(t, to_numpy=True)

def fitzhugh_nagumo(y, t, a, gamma, epsilon, Iext):
    v, w = y
    dvdt=v * (v - a) * (1 - v) - w + Iext
    dwdt = epsilon * (v - gamma * w)
    return [dvdt, dwdt]

y0 = [0.0, 0.0]
solution = odeint(fitzhugh_nagumo, y0, t, args=(a, gamma, epsilon, Iext))
v_num = solution[:,0]
w_num = solution[:,1]

print(internals)

fig = plt.figure(figsize=(12, 5))
ax1, ax2 = fig.subplots(1, 2)
ax1.plot(t, v_net, label='NN solution of v')
ax1.plot(t, v_num, '.', label='numerical solution of v')
ax1.plot(t, w_net, label='NN solution of w')
ax1.plot(t, w_num, '.', label='numerical solution of w')
ax1.set_ylabel('')
ax1.set_xlabel('t')
ax1.set_title('Comparing solutions')
ax1.legend()
ax2.set_title('Error of ANN solution from numerical solution')
ax2.plot(t, v_net-v_num, label='v_net-v_num,')
ax2.plot(t, w_net-w_num, label='w_net-w_num')
ax2.set_ylabel('')
ax2.set_xlabel('t')
ax2.legend()
plt.show()


