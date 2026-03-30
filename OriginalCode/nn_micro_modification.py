import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ========================
# 1. Neural Network Architecture
# ========================
class HypoidSolver(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN for pressure patterns (51x251)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        # FNN for hypoid parameters (11 inputs)
        self.fc_params = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU()
        )
        # Modification predictor (5 outputs)
        self.mod_predictor = nn.Sequential(
            nn.Linear(32*12*62 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 5)
        )
    
    def forward(self, pressure, params):
        cnn_feat = self.cnn(pressure)
        param_feat = self.fc_params(params)
        combined = torch.cat([cnn_feat, param_feat], dim=1)
        return self.mod_predictor(combined)

# ========================
# 2. TCA Simulator (Placeholder)
# ========================
class TCASimulator:
    def __init__(self):
        # Replace with actual TCA logic
        self.ideal_pressure = torch.rand(1, 1, 51, 251) * 500  # Synthetic target
    
    def run(self, hypoid_params, modifications):
        """Returns synthetic pressure pattern based on modifications"""
        noise = torch.randn(1, 1, 51, 251) * 50
        pressure = self.ideal_pressure + 200 * modifications.mean() + noise
        return pressure.clamp(min=0)

# ========================
# 3. Reward Calculation
# ========================
def compute_reward(pressure):
    """Lower reward = better (penalize peaks, gradients, edge unloading)"""
    peak_penalty = torch.max(pressure)
    gradient_penalty = torch.std(pressure)
    edge_penalty = -torch.mean(pressure[..., 0:10, :])  # Penalize low edge pressure
    return peak_penalty + gradient_penalty + edge_penalty

# ========================
# 4. Policy Gradient Training Loop
# ========================
class OptimizationEngine:
    def __init__(self):
        self.policy_net = HypoidSolver()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.tca_sim = TCASimulator()
        self.exploration_noise = 0.2  # Start high, decay over time
    
    def train_episode(self, hypoid_params, init_pressure, n_steps=10):
        current_pressure = init_pressure.clone()
        
        for step in range(n_steps):
            # 1. Predict modifications with exploration
            with torch.no_grad():
                mods = self.policy_net(current_pressure, hypoid_params)
                mods += self.exploration_noise * torch.randn_like(mods)
            
            # 2. Run TCA simulation
            new_pressure = self.tca_sim.run(hypoid_params, mods)
            
            # 3. Compute reward
            reward = compute_reward(new_pressure)
            
            # 4. Policy gradient update
            self.optimizer.zero_grad()
            
            # Get "deterministic" modifications (without exploration)
            pred_mods = self.policy_net(current_pressure, hypoid_params)
            
            # Surrogate loss: Encourage modifications that reduced reward
            loss = -pred_mods.mean() * reward  # Simplified policy gradient
            
            loss.backward()
            self.optimizer.step()
            
            # Update state
            current_pressure = new_pressure
            self.exploration_noise *= 0.95  # Decay noise
            
        return current_pressure, reward

# ========================
# 5. Main Execution
# ========================
if __name__ == "__main__":
    # Initialize
    engine = OptimizationEngine()
    
    # Example hypoid gear parameters (normalized)
    hypoid_params = torch.rand(1, 11)  # [gear_crown_dia, spiral_angle, ...]
    init_pressure = torch.rand(1, 1, 51, 251) * 1000  # Initial pressure pattern
    
    # Optimization loop
    for episode in range(50):
        final_pressure, reward = engine.train_episode(hypoid_params, init_pressure)
        print(f"Episode {episode+1} | Final Reward: {reward.item():.1f}")

    # Save final policy
    torch.save(engine.policy_net.state_dict(), "hypoid_solver.pth")