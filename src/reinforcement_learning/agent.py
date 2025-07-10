"""
Module for the Reinforcement Learning agent responsible for re-ranking.
Includes the policy network and training logic (e.g., GRPO).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Any
import io
import logging

from configs import config
from src.utils.azure_blob_storage import AzureBlobStorage

class RankingPolicyNetwork(nn.Module):
    """Neural network to parameterize the ranking policy."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim) # Outputs scores for each document

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Produce scores for each candidate document given the state."""
        x = self.relu(self.fc1(state))
        scores = self.fc2(x)
        return scores

class RLAgent:
    """RL Agent implementing a ranking policy and training algorithm (e.g., GRPO)."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, lr: float = config.LEARNING_RATE):
        self.policy_network = RankingPolicyNetwork(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.output_dim = output_dim # Should match initial_top_k
        
        try:
            self.blob_storage = AzureBlobStorage(container_name=config.AZURE_CONTAINER_NAME)
        except ValueError as e:
            logging.error(f"Azure Blob Storage inicializálási hiba az RLAgent-ben: {e}")
            # Lehetővé tesszük a működést offline módban, de a mentés/betöltés nem fog működni.
            self.blob_storage = None

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select an action (document scores) based on the current policy.
        For stochastic policies, this would involve sampling. For deterministic, just forward pass.
        Let's assume deterministic for now, outputting scores directly.
        """
        if state is None or state.size == 0:
             # Handle invalid state, maybe return zero scores
             return np.zeros(self.output_dim, dtype=np.float32)

        state_tensor = torch.FloatTensor(state).unsqueeze(0) # Add batch dimension
        with torch.no_grad():
            scores = self.policy_network(state_tensor)
        return scores.squeeze(0).numpy() # Remove batch dimension

    def update(self, batch: List[Dict[str, Any]]):
        """
        Update the policy network based on a batch of experiences.
        The structure of 'batch' depends heavily on the chosen RL algorithm (GRPO, Policy Gradient, etc.).

        Args:
            batch: A list of experiences. Each experience should contain at least:
                   - state: The state (query + candidates)
                   - action: The scores/ranking produced by the policy
                   - reward: The reward obtained for that ranking (from expert eval)
                   - (Potentially other info needed by the algorithm, like baseline scores)
        """
        if config.RL_ALGORITHM == "GRPO":
            self._update_grpo(batch)
        elif config.RL_ALGORITHM == "PolicyGradient":
            self._update_policy_gradient(batch)
        else:
            raise NotImplementedError(f"RL Algorithm '{config.RL_ALGORITHM}' not implemented.")

    def _update_grpo(self, batch: List[Dict[str, Any]]):
        """
        Update policy using Generalized Reward Policy Optimization (GRPO).
        This is complex and requires careful implementation based on the GRPO paper.
        It typically involves comparing pairs of rankings sampled from the policy
        and using the difference in rewards to update the policy.
        """
        # --- Placeholder for GRPO Logic ---
        # 1. Process the batch to get states, actions (rankings), rewards.
        # 2. For each state, potentially sample multiple rankings from the current policy.
        # 3. Compare pairs of rankings (e.g., ranking 'a' vs ranking 'b').
        # 4. Calculate the probability ratio of generating ranking 'a' vs 'b'.
        # 5. Use the reward difference (reward(a) - reward(b)) and the probability ratio
        #    to compute the loss according to the GRPO objective.
        # 6. Backpropagate the loss and update the network.
        print("Warning: GRPO update logic is complex and needs full implementation.")

        # Example structure (highly simplified):
        loss = torch.tensor(0.0, requires_grad=True)
        num_pairs = 0
        for i in range(len(batch)):
             for j in range(i + 1, len(batch)): # Simplified pairwise comparison within batch
                 # Assume batch items have 'state', 'action_scores', 'reward'
                 item_i = batch[i]
                 item_j = batch[j]

                 # This comparison assumes items i and j are comparable (e.g., different rankings for the same state)
                 # GRPO typically requires sampling multiple trajectories/rankings per state.
                 if np.array_equal(item_i['state'], item_j['state']): # Only compare if states match
                     reward_i = item_i['reward']
                     reward_j = item_j['reward']

                     if reward_i != reward_j: # Only update if rewards differ
                         # Calculate log probabilities of the actions (scores) under the current policy
                         state_tensor = torch.FloatTensor(item_i['state']).unsqueeze(0)
                         scores_i = self.policy_network(state_tensor)
                         scores_j = self.policy_network(state_tensor) # Re-run for grad? Or use stored scores?

                         # Need a way to get log_prob of the specific ranking/scores that were taken
                         # This depends on how scores are converted to rankings (e.g., Plackett-Luce model)
                         # log_prob_i = self._calculate_log_prob(state_tensor, item_i['action_scores']) # Placeholder
                         # log_prob_j = self._calculate_log_prob(state_tensor, item_j['action_scores']) # Placeholder

                         # Simplified loss based on reward difference (needs proper GRPO formulation)
                         # pair_loss = - (log_prob_i - log_prob_j) * (reward_i - reward_j) # Incorrect, just illustrative
                         # loss = loss + pair_loss
                         num_pairs += 1

        if num_pairs > 0:
             # average_loss = loss / num_pairs
             # self.optimizer.zero_grad()
             # average_loss.backward()
             # self.optimizer.step()
             pass # Skip actual update in placeholder

    def _update_policy_gradient(self, batch: List[Dict[str, Any]]):
        """Update policy using a basic Policy Gradient approach (e.g., REINFORCE for ranking)."""
        # --- Placeholder for Policy Gradient Logic ---
        # 1. Process batch: states, actions (rankings/scores), rewards.
        # 2. Calculate log probabilities of the actions taken under the current policy.
        # 3. Compute loss: - sum(log_prob * (reward - baseline))
        # 4. Backpropagate and update.
        print("Warning: Policy Gradient update logic needs implementation.")
        loss = torch.tensor(0.0, requires_grad=True)
        baseline = np.mean([item['reward'] for item in batch]) # Simple baseline

        for experience in batch:
            state = experience['state']
            action_scores = experience['action'] # The scores produced
            reward = experience['reward']

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            current_scores = self.policy_network(state_tensor)

            # Need a way to calculate log_prob of the action (scores)
            # If scores are means of a Gaussian, log_prob is Gaussian PDF
            # If scores are logits for categorical (e.g., Plackett-Luce), need different calculation
            # log_prob = self._calculate_log_prob(current_scores, action_scores) # Placeholder

            # advantage = reward - baseline
            # loss = loss - log_prob * advantage # Accumulate loss

        if len(batch) > 0:
            # average_loss = loss / len(batch)
            # self.optimizer.zero_grad()
            # average_loss.backward()
            # self.optimizer.step()
            pass # Skip actual update in placeholder

    def _calculate_log_prob(self, scores_distribution, actual_scores):
        """Placeholder: Calculate log probability of observed scores given the policy's output distribution."""
        # This depends heavily on the interpretation of the policy network's output (scores)
        # E.g., if scores are means of Gaussians with fixed std dev:
        # dist = torch.distributions.Normal(scores_distribution, std_dev)
        # return dist.log_prob(torch.FloatTensor(actual_scores)).sum()
        raise NotImplementedError

    def save(self):
        """Save the policy network state to Azure Blob Storage."""
        if not self.blob_storage:
            logging.warning("Nincs Azure Blob Storage kliens, a mentés kimarad.")
            return
            
        filepath = config.BLOB_RL_AGENT_PATH
        logging.info(f"RL Agent mentése ide: {filepath}")
        try:
            buffer = io.BytesIO()
            torch.save(self.policy_network.state_dict(), buffer)
            buffer.seek(0)
            self.blob_storage.upload_data(buffer.getvalue(), filepath)
            logging.info(f"RL Agent sikeresen mentve ide: {filepath}")
        except Exception as e:
            logging.error(f"Hiba az RL Agent mentésekor: {e}", exc_info=True)

    def load(self):
        """Load the policy network state from Azure Blob Storage."""
        if not self.blob_storage:
            logging.warning("Nincs Azure Blob Storage kliens, a betöltés kimarad.")
            return

        filepath = config.BLOB_RL_AGENT_PATH
        logging.info(f"RL Agent betöltése innen: {filepath}")
        try:
            data = self.blob_storage.download_data(filepath)
            buffer = io.BytesIO(data)
            self.policy_network.load_state_dict(torch.load(buffer))
            self.policy_network.eval()
            logging.info(f"RL Agent sikeresen betöltve innen: {filepath}")
        except Exception as e:
            # Ha a fájl nem létezik, az is Exception-t dob, amit itt kezelünk.
            logging.warning(f"Nem sikerült betölteni az RL Agentet innen: {filepath}. Új modellel indulunk. Hiba: {e}")

