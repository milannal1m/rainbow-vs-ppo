import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.output(x)
    
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        # 2 Streams and Output Layers
        # Value Stream
        self.fc_value = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

        # Advantages Stream
        self.fc_advantages = nn.Linear(hidden_dim, hidden_dim)
        self.advantages = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        # Value Calculation
        v = F.relu(self.fc_value(x))
        V = self.value(v)

        # Advantages Calculation
        a = F.relu(self.fc_advantages(x))
        A = self.advantages(a)

        # Calc Q 
        Q = V + A - torch.mean(A, dim=1, keepdim=True)
        
        return Q
    
if __name__ == "__main__":
    state_dim = 12
    action_dim = 2
    model = DQN(state_dim, action_dim)
    print(model)
    state = torch.randn(10, state_dim)
    q_values = model(state)
    print(q_values)