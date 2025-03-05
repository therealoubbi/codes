import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# FitzHugh-Nagumo model equations for generating synthetic data
def fitzhugh_nagumo(y, t, a, gamma, epsilon, Iext):
    v, w = y
    dvdt=v * (v - a) * (1 - v) - w + Iext
    dwdt = epsilon * (v - gamma * w)
    return [dvdt, dwdt]

# Generate synthetic data
def generate_data(a=0.7, gamma=0.8, epsilon=0.2, Iext=0.75, t_max=1000, n_points=1000):
    t = np.linspace(0, t_max, n_points)
    y0 = [0.0, 0.0]
    solution = odeint(fitzhugh_nagumo, y0, t, args=(a, gamma, epsilon, Iext))
    t_tensor = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)
    solution_tensor = torch.tensor(solution, dtype=torch.float32)
    return t_tensor, solution_tensor

# Neural Network for FitzHugh-Nagumo
class FitzHughNagumoModel(nn.Module):
    def __init__(self):
        super(FitzHughNagumoModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 126),
            nn.Tanh(),
            nn.Linear(126, 126),
            nn.Tanh(),
            nn.Linear(126, 126),
            nn.Tanh(),
            nn.Linear(126, 126),
            nn.Tanh(),            
            nn.Linear(126, 2)
        )
        
    def forward(self, t):
        return self.net(t)

# Physics-informed loss: FitzHugh-Nagumo dynamics
def physics_loss(y_pred, t_tensor, a=0.7, gamma=0.8, epsilon=0.2, Iext=0.75):
    v, w = y_pred[:, 0], y_pred[:, 1]
    v_t = torch.autograd.grad(v.sum(), t_tensor, create_graph=True)[0]
    w_t = torch.autograd.grad(w.sum(), t_tensor, create_graph=True)[0]
    f_v = v * (v - a) * (1 - v) - w + Iext
    f_w = epsilon * (v - gamma * w)
    return torch.mean((v_t-f_v)**2)+torch.mean((w_t-f_w)**2)

# Training the model
def train_model(t_data, y_data, epochs=1000, learning_rate=1e-3):
    model = FitzHughNagumoModel()
    gamma = torch.nn.Parameter(torch.ones(1, requires_grad=True))
    alpha = torch.nn.Parameter(torch.ones(1, requires_grad=True))
    beta = torch.nn.Parameter(torch.ones(1, requires_grad=True))
    optimizer = torch.optim.Adam(list(model.parameters())+[gamma]+[alpha]+[beta], lr=1e-3)

    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        t_data.requires_grad = True  # Enable gradient computation for t_data
        y_pred = model(t_data)
        loss_data = criterion(y_pred, y_data)
        loss_physics = physics_loss(y_pred, t_data)
        loss_boundary = torch.mean((y_pred[0])**2) + torch.mean((y_pred[1])**2)
        loss = alpha * loss_data + gamma * loss_physics + beta * loss_boundary
        loss.backward()
        optimizer.step()

        # Print weights and loss
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}, Loss Data: {loss_data.item()}, Loss Physics: {loss_physics.item()}')
            

    return model


# Plotting and main execution are the same as before
import random

# Splitting data into training and testing sets
def split_data(tfn, yfn, train_ratio=0.75):
    # Get the total number of data points
    num_points = tfn.shape[0]
    # Shuffle indices
    indices = list(range(num_points))
    random.shuffle(indices)
    # Calculate the number of points for training
    num_train_points = int(train_ratio * num_points)
    # Split indices into training and testing indices
    train_indices = indices[:num_train_points]
    test_indices = indices[num_train_points:]
    # Split data based on indices
    t_train = tfn[train_indices]
    t_test = tfn[test_indices]
    y_train = yfn[train_indices]
    y_test = yfn[test_indices]
    return t_train, y_train, t_test, y_test

# Generating data
tfn, yfn = generate_data()

# Splitting data
t_data, y_data, t_test, y_test = split_data(tfn, yfn)

# Training the model



model = train_model(t_data, y_data)
# Plotting predictions - Use the same function as before
def plot_predictions(t_test,y_test, model):
    # Ensure no gradient is computed for plotting
    with torch.no_grad():
        t_test.requires_grad = False
        y_pred = model(t_test).numpy()  # Get model predictions
        
    # Convert tensors to numpy arrays for plotting
    t_numpy = t_test.numpy().flatten()
    y_true = y_test.numpy()

    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot v component
    plt.subplot(2, 1, 1)
    plt.plot(t_numpy, y_true[:,0], 'r', label='True v')
    plt.plot(t_numpy, y_pred[:,0], 'b--', label='Predicted v')
    plt.xlabel('Time')
    plt.ylabel('v')
    plt.legend()
    plt.title('FitzHugh-Nagumo Model: True vs. Predicted Dynamics')
    
    # Plot w component
    plt.subplot(2, 1, 2)
    plt.plot(t_numpy, y_true[:, 1], 'g', label='True w')
    plt.plot(t_numpy, y_pred[:, 1], 'k--', label='Predicted w')
    plt.xlabel('Time')
    plt.ylabel('w')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Assuming t_data and y_data are already defined and the model is trained
plot_predictions(tfn, yfn, model)
