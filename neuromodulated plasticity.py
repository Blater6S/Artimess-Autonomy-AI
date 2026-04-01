# Author : P.P. Chanchal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import asyncio
import logging
import heapq
from collections import deque, namedtuple
from typing import List, Tuple, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [AI_BRAIN] %(message)s'
)

StateAction = namedtuple('StateAction', ['state', 'action', 'reward', 'next_state'])

class MultimodalFusionLayer(nn.Module):
    """
    Fuses vectors from Text (768), Image (1284), and Audio (768) into a common hidden state.
    """
    def __init__(self, hidden_size=512):
        super().__init__()
        self.text_proj = nn.Linear(768, hidden_size)
        self.image_proj = nn.Linear(1284, hidden_size) 
        self.audio_proj = nn.Linear(768, hidden_size)
        
        self.fusion_attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, text_vec=None, image_vec=None, audio_vec=None):
        device = next(self.parameters()).device
        batch_size = 1
        
        # Handle missing inputs with zero vectors
        if text_vec is None: text_vec = torch.zeros(batch_size, 768).to(device)
        if image_vec is None: image_vec = torch.zeros(batch_size, 1284).to(device)
        if audio_vec is None: audio_vec = torch.zeros(batch_size, 768).to(device)
        
        # Ensure 2D
        if text_vec.dim() == 1: text_vec = text_vec.unsqueeze(0)
        if image_vec.dim() == 1: image_vec = image_vec.unsqueeze(0)
        if audio_vec.dim() == 1: audio_vec = audio_vec.unsqueeze(0)

        # Project
        t = self.text_proj(text_vec).unsqueeze(1)
        i = self.image_proj(image_vec).unsqueeze(1)
        a = self.audio_proj(audio_vec).unsqueeze(1)
        
        # Concatenate [batch, 3, hidden]
        seq = torch.cat([t, i, a], dim=1)
        
        # Self-Attention
        attn_out, _ = self.fusion_attention(seq, seq, seq)
        
        # Pool (Mean)
        fused = attn_out.mean(dim=1)
        
        return self.norm(self.output_proj(fused))

class RLDecisionModule(nn.Module):
    """
    Actor-Critic Network for decision making.
    """
    def __init__(self, state_dim=512, action_dim=7, hidden_dim=256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value

class AStarPlanner:
    """
    A* Planner for long-term goal seeking in the latent space.
    """
    def __init__(self, heuristic_weight=1.0):
        self.heuristic_weight = heuristic_weight
        
    def heuristic(self, state: torch.Tensor, goal: torch.Tensor) -> float:
        return 1 - F.cosine_similarity(state, goal, dim=-1).item()
    
    def plan(self, current_state: torch.Tensor, goal_state: torch.Tensor, 
             action_space: List[Dict], max_depth=10) -> List[str]:
        # Simplified A* for demonstration
        # In a real latent space, transitions are learned models.
        # Here we simulate a path.
        return ["plan_step_1", "plan_step_2"]

class UltimateRLNeuromodulatedAI(nn.Module):
    """
    The Core Brain. Combines Fusion, RL, and Plasticity.
    """
    def __init__(self, input_size=768, hidden_size=512):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_size = hidden_size
        
        # Components
        self.fusion = MultimodalFusionLayer(hidden_size)
        
        self.action_space = [
            {'name': 'analyze_text', 'reward': 1.0},
            {'name': 'process_image', 'reward': 0.8},
            {'name': 'listen_audio', 'reward': 0.9},
            {'name': 'generate_speech', 'reward': 1.2},
            {'name': 'create_image', 'reward': 1.1},
            {'name': 'plan_sequence', 'reward': 1.5},
            {'name': 'retrieve_memory', 'reward': 0.7},
        ]
        
        self.rl_decision = RLDecisionModule(hidden_size, action_dim=len(self.action_space))
        self.a_star = AStarPlanner()
        
        # Memory & Learning
        self.replay_buffer = deque(maxlen=10000)
        self.loss_history = deque(maxlen=100)
        self.plasticity_rate = 0.01  # Adaptive
        
    def select_action(self, state: torch.Tensor, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.choice(len(self.action_space))
        
        probs, _ = self.rl_decision(state)
        return torch.multinomial(probs, 1).item()
    
    def store_experience(self, state, action, reward, next_state):
        self.replay_buffer.append(StateAction(state, action, reward, next_state))
    
    def rl_update(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return torch.tensor(0.0)
            
        batch = np.random.choice(len(self.replay_buffer), batch_size)
        batch = [self.replay_buffer[i] for i in batch]
        
        states = torch.stack([exp.state for exp in batch])
        actions = torch.tensor([exp.action for exp in batch]).unsqueeze(1)
        rewards = torch.tensor([exp.reward for exp in batch]).unsqueeze(1)
        next_states = torch.stack([exp.next_state for exp in batch])
        
        # Critic
        _, current_values = self.rl_decision(states)
        _, next_values = self.rl_decision(next_states)
        targets = rewards + 0.99 * next_values.detach()
        critic_loss = F.mse_loss(current_values.gather(1, actions), targets)
        
        # Actor
        actor_probs, _ = self.rl_decision(states)
        actor_loss = -torch.log(actor_probs.gather(1, actions)) * (targets - current_values.detach())
        actor_loss = actor_loss.mean()
        
        total_loss = actor_loss + critic_loss
        return total_loss

    def sleep_cycle(self):
        """
        Consolidate memories and adjust plasticity (Self-Modification).
        """
        if not self.loss_history: return
        
        avg_loss = np.mean(self.loss_history)
        
        # Adaptive Plasticity: If loss is high, increase plasticity (learn faster)
        # If loss is low, decrease plasticity (stabilize)
        if avg_loss > 1.0:
            self.plasticity_rate = min(self.plasticity_rate * 1.1, 0.1)
            logging.info(f"Sleep: High loss ({avg_loss:.2f}). Increasing plasticity to {self.plasticity_rate:.4f}")
        else:
            self.plasticity_rate = max(self.plasticity_rate * 0.9, 0.001)
            logging.info(f"Sleep: Low loss ({avg_loss:.2f}). Stabilizing. Plasticity: {self.plasticity_rate:.4f}")
            
        # Clear history for next cycle
        self.loss_history.clear()

# Wrapper for compatibility with main.py
class AutonomousNeuromodulatedAI:
    def __init__(self, input_size=768, hidden_size=512):
        self.model = UltimateRLNeuromodulatedAI(input_size, hidden_size)
        self.device = self.model.device
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.steps = 0
        
    def train_step(self, vector, target):
        self.steps += 1
        
        # Adapt vector to fusion input
        size = vector.size(-1)
        text_vec = vector if size == 768 else None
        image_vec = vector if size == 1284 else None
        
        # Forward
        fused_state = self.model.fusion(text_vec, image_vec, None)
        
        # Action
        action_idx = self.model.select_action(fused_state)
        action = self.model.action_space[action_idx]
        
        # Simulate Reward (Self-Supervised: did we predict the target?)
        # Here we just use a dummy reward for the loop
        reward = 1.0 
        
        # Store
        self.model.store_experience(fused_state, action_idx, reward, fused_state)
        
        # Train
        loss = self.model.rl_update()
        
        # Update weights
        self.optimizer.zero_grad()
        if loss.requires_grad:
            loss.backward()
            self.optimizer.step()
        
        # Track loss
        self.model.loss_history.append(loss.item())
        
        # Sleep Cycle every 100 steps
        if self.steps % 100 == 0:
            self.model.sleep_cycle()
        
        return {
            'avg_loss': loss.item(),
            'avg_accuracy': 0.0,
            'action': action['name']
        }

if __name__ == "__main__":
    # Test
    ai = AutonomousNeuromodulatedAI()
    print("Brain initialized.")
