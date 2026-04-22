import numpy as np

def preprocess_state(state):
    """
    Convierte el estado del entorno a un formato numérico para la red neuronal
    """
    player_sum, dealer_card, usable_ace = state
    
    return np.array([
        float(player_sum),
        float(dealer_card),
        float(usable_ace)
    ])

from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)