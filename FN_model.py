import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# FitzHugh-Nagumo equations
def FHN(t, Y, a, epsilon, gamma, I_ext):
    v, w = Y
    dvdt = v * (v - a) * (1 - v) - w + I_ext
    dwdt = epsilon * (v - gamma * w)
    return [dvdt, dwdt]

# Initial conditions
V0 = 0.0
W0 = 0.0
Y0 = [V0, W0]

# Time vector
t_span = [0, 100]  # Start and end time
t_eval = np.linspace(0, 100, 10000)  # Time points where the solution is computed

# Parameter sets
parameter_sets = [
    (0.7, 0.2, 0.8, 0.4),
    (0.7, 0.2, 0.8, 0.7),
    (0.7, 0.2, 0.8, 0.8),
    (0.7, 0.2, 0.8, 1)
]

# Create a figure and axes for plotting
fig, axs = plt.subplots(4, 2, figsize=(12, 16))

# Solve the differential equations for each parameter set and plot results
for idx, (a, epsilon, gamma, I_ext) in enumerate(parameter_sets):
    sol = solve_ivp(FHN, t_span, Y0, args=(a, epsilon, gamma, I_ext), t_eval=t_eval, method='RK45')
    v = sol.y[0, :]
    w = sol.y[1, :]

    # Membrane potential V over time
    axs[idx, 0].plot(sol.t, v, label=f'V (a={a}, ε={epsilon}, γ={gamma})')
    axs[idx, 0].set_title(f'Set {idx + 1}: Membrane Potential')
    axs[idx, 0].set_ylabel('V')
    axs[idx, 0].legend()

    # Recovery variable W over time
    axs[idx, 1].plot(sol.t, w, label=f'W (a={a}, ε={epsilon}, γ={gamma})')
    axs[idx, 1].set_title(f'Set {idx + 1}: Recovery Variable')
    axs[idx, 1].set_ylabel('W')
    axs[idx, 1].legend()

    # For the last set, set the X labels
    if idx == len(parameter_sets) - 1:
        axs[idx, 0].set_xlabel('Time (ms)')
        axs[idx, 1].set_xlabel('Time (ms)')

# Add an additional plot for phase plane (V vs W) for each parameter set
fig_phase, ax_phase = plt.subplots(figsize=(6, 6))
for a, epsilon, gamma, I_ext in parameter_sets:
    sol = solve_ivp(FHN, t_span, Y0, args=(a, epsilon, gamma, I_ext), t_eval=t_eval, method='RK45')
    v = sol.y[0, :]
    w = sol.y[1, :]
    ax_phase.plot(v, w, label=f'a={a}, ε={epsilon}, γ={gamma}')

ax_phase.set_title('Phase Plane (V vs W)')
ax_phase.set_xlabel('V')
ax_phase.set_ylabel('W')
ax_phase.legend()
ax_phase.grid(True)

# Show plots
plt.tight_layout()
plt.show()
