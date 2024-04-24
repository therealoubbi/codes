import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math

# Constants for the Beeler-Reuter model
C = 0.01
g_Na = 0.04
g_NaC = 3e-05
E_Na = 50
g_s = 0.0009


# Define the stimulus protocol parameters
IstimStart = 10  # ms
IstimEnd = 50000  # ms
IstimAmplitude = 0.5  # uA_per_mm2
IstimPeriod = 1000  # ms
IstimPulseDuration = 1  # ms




# Gating variables' rate functions
import numpy as np

def alpha_m(V=0):
    return (-47 - V) / (-1 + 0.009095277101695816 * np.exp(-0.1 * V))

def beta_m(V=0):
    return 0.7095526727489909 * np.exp(-0.056 * V)

def alpha_h(V=0):
    return 5.497962438709065e-10 * np.exp(-0.25 * V)

def beta_h(V=0):
    return 1.7 / (1 + 0.1580253208896478 * np.exp(-0.082 * V))

def alpha_j(V=0):
    return 1.8690473007222892e-10 * np.exp(-0.25 * V) / (1 + 1.6788275299956603e-07 * np.exp(-0.2 * V))

def beta_j(V=0):
    return 0.3 / (1 + 0.040762203978366204 * np.exp(-0.1 * V))

def alpha_d(V=0):
    return 0.095 * np.exp(1 / 20 - V / 100) / (1 + 1.4332881385696572 * np.exp(-0.07199424046076314 * V))

def beta_d(V=0):
    return 0.07 * np.exp(-44 / 59 - V / 59) / (1 + np.exp(11 / 5 + V / 20))

def alpha_f(V=0):
    return 0.012 * np.exp(-28 / 125 - V / 125) / (1 + 66.5465065250986 * np.exp(0.14992503748125938 * V))

def beta_f(V=0):
    return 0.0065 * np.exp(-3 / 5 - V / 50) / (1 + np.exp(-6 - V / 5))

def alpha_x1(V=0):
    return 0.031158410986342627 * np.exp(0.08264462809917356 * V) / (1 + 17.41170806332765 * np.exp(0.05714285714285714 * V))

def beta_x1(V=0):
    return 0.0003916464405623223 * np.exp(-0.05998800239952009 * V) / (1 + np.exp(-4 / 5 - V / 25))

# Compute derivatives function, corrected for indexing and including all ionic currents
def compute_derivatives(t0, y):
    V, m, h, j, d, f, x1, Ca_i = y
    dy = np.zeros((8,))
    

    if (IstimStart <= t0 <= IstimEnd) and ((t0 - IstimStart) % IstimPeriod) <= IstimPulseDuration:
        Istim_current = IstimAmplitude
    else:
        Istim_current = 0

    # Use Istim_current in the dy[0] equation
    
    
    # Ionic currents
    I_K1 = 0.0035 * (4.6000000000000005 + 0.2 * V) / (
        1 - 0.39851904108451414 * np.exp(-0.04 * V)
    ) + 0.0035 * (-4 + 119.85640018958804 * np.exp(0.04 * V)) / (
        8.331137487687693 * np.exp(0.04 * V) + 69.4078518387552 * np.exp(0.08 * V)
    )
    I_x1 = (
        0.0019727757115328517
        * (-1 + 21.75840239619708 * np.exp(0.04 * V))
        * np.exp(-0.04 * V)
        * x1
    )
    I_Na = (g_Na * m**3 * h * j + g_NaC) * (V - E_Na)   
    E_s = -82.3 - 13.0287 * np.log(0.001 * Ca_i)
    i_s = g_s * (-E_s + V) * d * f

    # Differential equations for the membrane potential and gating variables
    dy[0] = -((I_Na + I_x1 + I_K1+i_s)-Istim_current ) / C
    dy[1] = alpha_m(V) * (1 - m) - beta_m(V) * m
    dy[2] = alpha_h(V) * (1 - h) - beta_h(V) * h
    dy[3] = alpha_j(V) * (1 - j) - beta_j(V) * j
    dy[4] = alpha_d(V) * (1 - d) - beta_d(V) * d
    dy[5] = alpha_f(V) * (1 - f) - beta_f(V) * f
    dy[6] = alpha_x1(V) * (1 - x1) - beta_x1(V) * x1
    dy[7] =  7.000000000000001e-06 - 0.07 * Ca_i - 0.01 * i_s # Calcium dynamics

    return dy

# Initial conditions y0= [ V, m, h, j, d, f, x1, Ca_i] 
y0 = [-84.624, 0.011, 0.988, 0.975, 0.003, 0.994, 0.0001, 0.0001]

# Integrate ODE

t_span = [0, 1000]  # For example, simulate for 500 ms
t_eval = np.linspace(t_span[0], t_span[1], 10000) 

solution = solve_ivp(compute_derivatives, t_span, y0, t_eval=t_eval)

# Plot the results for the membrane potential V
plt.plot(solution.t, solution.y[0])
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Beeler-Reuter Model Simulation')
plt.show()
