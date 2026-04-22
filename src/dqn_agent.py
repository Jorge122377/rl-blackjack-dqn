import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
import random
import torch


def select_action(model, state, epsilon):
    """
    Selecciona una acción usando estrategia epsilon-greedy
    """
    if random.random() < epsilon:
        return random.randint(0, 1)  # explorar
    else:
        with torch.no_grad():
            q_values = model(state)
            return torch.argmax(q_values).item()  # explotar