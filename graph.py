import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from sklearn.datasets import make_moons
from scipy.stats import gaussian_kde

# Copy of model definition (keeps graph.py self-contained)
class DenoiseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
    def forward(self, x, t):
        t = t / 1000.0
        t_embed = t.unsqueeze(-1)
        h = torch.cat([x, t_embed], dim=1)
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        return self.fc3(h)

# load data
X, _ = make_moons(n_samples=2000, noise=0.05)
X = torch.tensor(X, dtype=torch.float32)

# plotting grid
x_min, x_max = -2.5, 2.5
y_min, y_max = -1.5, 1.5
GRID_SIZE = 12
x_coords = np.linspace(x_min, x_max, GRID_SIZE)
y_coords = np.linspace(y_min, y_max, GRID_SIZE)
X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
GRID_POINTS = torch.tensor(np.vstack([X_grid.ravel(), Y_grid.ravel()]).T, dtype=torch.float32)

# instantiate and load model
model = DenoiseMLP()
state = torch.load("denoise_model_final.pth", map_location='cpu')
model.load_state_dict(state)
model.eval()

# Define schedules (must match diffusion.py)
T = 1000
beta = torch.linspace(1e-4, 0.02, T)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# Vector field plots (example timesteps)
steps = [999, 500, 200, 50, 0]
fig, axes = plt.subplots(1, len(steps), figsize=(3 * len(steps), 4), sharex=True, sharey=True)
for i, t in enumerate(steps):
    t_batch = torch.full((GRID_POINTS.shape[0],), t, dtype=torch.float32)
    with torch.no_grad():
        pred_noise = model(GRID_POINTS, t_batch)
    alpha_t = alpha[t]
    alpha_bar_t = alpha_bar[t]
    beta_t = beta[t]
    mu_tilde = (1 / torch.sqrt(alpha_t)) * (GRID_POINTS - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise)
    denoise_vec = (mu_tilde - GRID_POINTS).numpy()
    U = denoise_vec[:, 0].reshape(GRID_SIZE, GRID_SIZE)
    V = denoise_vec[:, 1].reshape(GRID_SIZE, GRID_SIZE)
    ax = axes[i]
    ax.quiver(X_grid, Y_grid, U, V, color='blue', angles='xy', scale_units='xy', scale=0.2, width=0.005)
    ax.set_title(f't={t}')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig(f"noise_vector_field_from_saved_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", dpi=300)
plt.close(fig)

# Experiment: plot trajectories from single starting point (one run)
N_TRACK = 1
tracked_x_initial = torch.randn(N_TRACK, 2)

@torch.no_grad()
def sample_and_track(model, x_start):
    trajectory = torch.zeros(T + 1, x_start.shape[0], 2)
    x = x_start.clone()
    trajectory[T] = x.clone()
    for t in reversed(range(T)):
        t_batch = torch.full((x.shape[0],), t, dtype=torch.float32)
        pred_noise = model(x, t_batch)
        alpha_t = alpha[t]
        alpha_bar_t = alpha_bar[t]
        beta_t = beta[t]
        mu_tilde = 1/torch.sqrt(alpha_t) * (x - (1-alpha_t)/torch.sqrt(1-alpha_bar_t)*pred_noise)
        if t > 0:
            z = torch.randn_like(x)
            x = mu_tilde + torch.sqrt(beta_t) * z
        else:
            x = mu_tilde
        trajectory[t] = x.clone()
    return x, trajectory

final, full_traj = sample_and_track(model, tracked_x_initial)

# plot multi-step and stochasticity experiments (reuse code from diffusion.py but simplified)
# Multi-step
steps_to_plot = [T, 750, 500, 250, 0]
fig_multi, axes_multi = plt.subplots(1, len(steps_to_plot), figsize=(3 * len(steps_to_plot), 4), sharex=True, sharey=True)
# compute kde
kde = gaussian_kde(np.vstack([X[:,0].numpy(), X[:,1].numpy()]))
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

for k, t_step in enumerate(steps_to_plot):
    ax = axes_multi[k]
    path_indices = list(range(T, t_step - 1, -1))
    path = full_traj[path_indices, 0]
    x_i = path[:,0].numpy(); y_i = path[:,1].numpy()
    ax.plot(x_i, y_i, color='gray', alpha=0.25, linewidth=1)
    head = full_traj[t_step, 0]
    ax.plot(head[0].numpy(), head[1].numpy(), 'o', color='C0', markersize=6, markeredgecolor='black', markeredgewidth=0.6, zorder=3)
    ax.contourf(xx, yy, zz, levels=10, cmap='Greys', alpha=0.35, zorder=1)
    ax.set_title(f't = {t_step}')
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig(f"diffusion_trajectories_multistep_from_saved_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", dpi=300)
plt.close(fig_multi)

# Experiment 2: repeated runs from same starting point
n_runs = 5
repeated_trajs = []
for r in range(n_runs):
    _, traj_run = sample_and_track(model, tracked_x_initial)
    repeated_trajs.append(traj_run.cpu())

fig_exp2, axes_exp2 = plt.subplots(1, n_runs, figsize=(3 * n_runs, 4), sharex=True, sharey=True)
for r in range(n_runs):
    ax = axes_exp2[r]
    traj = repeated_trajs[r][:, 0, :]
    x_r = traj[:, 0].numpy(); y_r = traj[:, 1].numpy()
    ax.plot(x_r, y_r, color='C{}'.format(r % 10), alpha=0.8, linewidth=0.8)
    ax.plot(x_r[0], y_r[0], marker='X', color='white', markersize=7, markeredgecolor='C{}'.format(r % 10), markeredgewidth=1.2, zorder=6)
    ax.plot(x_r[-1], y_r[-1], 'o', color='C{}'.format(r % 10), markersize=7, markeredgecolor='black', markeredgewidth=0.8, zorder=7)
    ax.contourf(xx, yy, zz, levels=10, cmap='Greys', alpha=0.35, zorder=1)
    ax.set_title(f'Run {r+1}'); ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig(f"diffusion_stochasticity_exp2_from_saved_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", dpi=300)
plt.close(fig_exp2)

print('Graphs generated from saved model.')
