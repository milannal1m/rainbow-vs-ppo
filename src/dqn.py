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
    
class CNNDQN(nn.Module):
    # Based on https://github.com/yenchenlin/DeepLearningFlappyBird/blob/master/deep_q_network.py
    def __init__(self, in_channels, action_dim, hidden_dim=512):
        super(CNNDQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        #self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        #self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        #If i ever uncomment i need to change the input layers
        self.fc1   = nn.Linear(1600, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        #x = self.pool2(x)
        x = F.relu(self.conv3(x))
        #x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.output(x)


NETWORK_REGISTRY = {
    "dqn":         DQN,
    "dueling_dqn": DuelingDQN,
    "cnn_dqn":     CNNDQN,
}


def optimize(batch, policy_dqn, target_dqn, optimizer, loss_fn, discount_factor_g, enable_double_dqn, device):
    states, actions, new_states, rewards, terminations = zip(*batch)

    states = torch.stack(states)
    actions = torch.stack(actions)
    new_states = torch.stack(new_states)
    rewards = torch.stack(rewards)
    terminations = torch.tensor(terminations, dtype=torch.bool).to(device)

    with torch.no_grad():
        if enable_double_dqn:
            best_actions = policy_dqn(new_states).max(dim=1)[1]
            target_q = rewards + (1 - terminations.float()) * discount_factor_g * \
                target_dqn(new_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
        else:
            target_q = rewards + (1 - terminations.float()) * discount_factor_g * \
                target_dqn(new_states).max(dim=1)[0]

    q_all = policy_dqn(states)
    current_q = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
    mean_max_q = q_all.max(dim=1)[0].mean().item()

    loss = loss_fn(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), 10)
    optimizer.step()
    return loss.item(), mean_max_q


if __name__ == "__main__":
    state_dim = 12
    action_dim = 2
    model = DQN(state_dim, action_dim)
    print(model)
    state = torch.randn(10, state_dim)
    q_values = model(state)
    print(q_values)