from jaxmarl.environments import MultiAgentEnv
from jaxmarl.environments import spaces
import jax.numpy as jnp
import chex
from typing import Dict, Tuple

class ActionObservableWrapper(MultiAgentEnv):
    """Wrapper that adds other agents' actions to each agent's observation."""
    
    def __init__(self, env: MultiAgentEnv):
        """Initialize wrapper with base environment."""
        self.env = env  # Store the original environment
        self._num_actions = env.action_space().n  # Get number of possible actions
        
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict, chex.Array]:
        """Reset environment and initialize action history."""
        obs, state = self.env.reset(key)  # Reset the base environment
        # Add zero actions for initial step since no previous actions exist
        state.last_actions = jnp.zeros(self.env.num_agents, dtype=jnp.int32)
        return obs, state
        
    def step(self, key: chex.PRNGKey, state: chex.Array, actions: Dict) -> Tuple[Dict, chex.Array, Dict, Dict, Dict]:
        """Execute environment step and augment observations with other agents' actions."""
        # Step the base environment
        obs, next_state, reward, done, info = self.env.step(key, state, actions)
        
        # Convert actions dictionary to array for easier handling
        # e.g., {"agent_0": 1, "agent_1": 3} -> [1, 3]
        action_array = jnp.array([actions[f"agent_{i}"] for i in range(self.env.num_agents)])
        
        # For each agent
        for i in range(self.env.num_agents):
            # Get all other agents' actions by removing agent i's action
            other_agents_actions = jnp.delete(action_array, i)
            # Convert actions to one-hot encoding
            # e.g., [1, 3] -> [[0,1,0,0,0,0], [0,0,0,1,0,0]] for 6 possible actions
            one_hot_actions = jax.nn.one_hot(other_agents_actions, self._num_actions)
            
            # Handle different observation types
            if len(obs[f"agent_{i}"].shape) == 3:  # Image observations (height, width, channels)
                # Create new layer for actions
                action_layer = jnp.zeros((*obs[f"agent_{i}"].shape[:2], self._num_actions))
                # Place actions in first row, first column of new layer
                action_layer = action_layer.at[0, 0, other_agents_actions].set(1)
                # Concatenate action layer with original observation
                obs[f"agent_{i}"] = jnp.concatenate([obs[f"agent_{i}"], action_layer], axis=-1)
            else:  # Vector observations
                # Simply concatenate flattened one-hot actions
                obs[f"agent_{i}"] = jnp.concatenate([obs[f"agent_{i}"], one_hot_actions.flatten()])
        
        # Store current actions for next step
        next_state.last_actions = action_array
        return obs, next_state, reward, done, info

    def observation_space(self, agent_id: str = "") -> spaces.Box:
        """Define the new observation space that includes other agents' actions."""
        base_space = self.env.observation_space(agent_id)
        
        if len(base_space.shape) == 3:  # Image observations
            # Add action channels to existing channels
            new_shape = (*base_space.shape[:2], base_space.shape[2] + self._num_actions)
        else:  # Vector observations
            # Add space for other agents' one-hot actions
            new_shape = (base_space.shape[0] + self._num_actions * (self.env.num_agents - 1),)
            
        return spaces.Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=base_space.dtype
        ) 