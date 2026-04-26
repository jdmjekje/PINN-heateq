# Physics Informed Neural Network for the 1D Heat Equation
Implemented a PINN to solve the 1D heat equation by embedding the PDE directly into the loss function.

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

L = 1.0
alpha = 0.01
T_final = 0.5

epochs = 2000
learning_rate = 1e-3

N_physics = 1000
N_ic = 100
N_bc = 100

target_times = [0.0, 0.05, 0.1, 0.2, 0.5]


# 2. Initial condition

def initial_condition_torch(x):
    return torch.exp(-100 * (x - 0.5) ** 2)


def initial_condition_np(x):
    return np.exp(-100 * (x - 0.5) ** 2)

# 3. PINN model


class PINN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)


model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_history = []

# 4. Training loop


for epoch in range(epochs):
    optimizer.zero_grad()

  
    # Physics loss: u_t = alpha u_xx
   
    x = torch.rand(N_physics, 1, requires_grad=True)
    t = torch.rand(N_physics, 1, requires_grad=True) * T_final

    u = model(x, t)

    u_t = torch.autograd.grad(
        u, t,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_x = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    physics_loss = torch.mean((u_t - alpha * u_xx) ** 2)

    # Initial condition loss: u(x,0)
  
    x_ic = torch.linspace(0, L, N_ic).view(-1, 1)
    t_ic = torch.zeros_like(x_ic)

    u_pred_ic = model(x_ic, t_ic)
    u_true_ic = initial_condition_torch(x_ic)

    ic_loss = torch.mean((u_pred_ic - u_true_ic) ** 2)


    # Boundary condition loss: u(0,t)=0, u(1,t)=0
 
    t_bc = torch.rand(N_bc, 1) * T_final

    x_left = torch.zeros_like(t_bc)
    x_right = torch.ones_like(t_bc) * L

    u_left = model(x_left, t_bc)
    u_right = model(x_right, t_bc)

    bc_loss = torch.mean(u_left ** 2) + torch.mean(u_right ** 2)

    # Total loss
   
    total_loss = physics_loss + 10 * ic_loss + 10 * bc_loss

    total_loss.backward()
    optimizer.step()

    loss_history.append(total_loss.item())

    if epoch % 200 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"Total Loss: {total_loss.item():.6e} | "
            f"Physics: {physics_loss.item():.2e} | "
            f"IC: {ic_loss.item():.2e} | "
            f"BC: {bc_loss.item():.2e}"
        )


# 5. Finite difference solution for comparison


Nx = 100
dx = L / (Nx - 1)
dt = 0.0001
Nt = int(T_final / dt)

r = alpha * dt / dx**2
print(f"\nFinite difference stability number r = {r:.4f}")

x_fd = np.linspace(0, L, Nx)
u_fd = initial_condition_np(x_fd)

snapshots = {}

for n in range(Nt + 1):
    current_time = n * dt

    for target in target_times:
        if abs(current_time - target) < dt / 2:
            snapshots[target] = u_fd.copy()

    u_new = u_fd.copy()

    for i in range(1, Nx - 1):
        u_new[i] = u_fd[i] + r * (u_fd[i + 1] - 2 * u_fd[i] + u_fd[i - 1])

    u_new[0] = 0
    u_new[-1] = 0

    u_fd = u_new


# 6. Plot PINN prediction


x_test = torch.linspace(0, L, Nx).view(-1, 1)

plt.figure(figsize=(8, 5))

for time in target_times:
    t_test = torch.ones_like(x_test) * time

    with torch.no_grad():
        u_test = model(x_test, t_test).numpy().flatten()

    plt.plot(x_test.numpy(), u_test, label=f"t = {time}")

plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("PINN Prediction for the 1D Heat Equation")
plt.legend()
plt.grid()
plt.show()



# 7. Plot training loss

plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("Training Loss")
plt.grid()
plt.show()

plt.figure(figsize=(9, 5))

for time in target_times:
    t_test = torch.ones_like(x_test) * time

    with torch.no_grad():
        u_pinn = model(x_test, t_test).numpy().flatten()

    plt.plot(x_fd, snapshots[time], "--", label=f"FD t = {time}")
    plt.plot(x_fd, u_pinn, label=f"PINN t = {time}")

plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("PINN vs Finite Difference Solution")
plt.legend()
plt.grid()
plt.show()

# 9. Error calculation

print("\nMean Squared Error compared to finite difference:")

for time in target_times:
    t_test = torch.ones_like(x_test) * time

    with torch.no_grad():
        u_pinn = model(x_test, t_test).numpy().flatten()

    u_fd = snapshots[time]
    mse = np.mean((u_pinn - u_fd) ** 2)

    print(f"Time {time:4.2f}: MSE = {mse:.6e}")
