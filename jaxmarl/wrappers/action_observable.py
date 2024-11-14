from jaxmarl.environments import MultiAgentEnv
from jaxmarl.environments import spaces
import jax.numpy as jnp
import chex
from typing import Dict, Tuple
import jax

class ActionObservableWrapper(MultiAgentEnv):
    """Wrapper that adds other agents' actions to each agent's observation."""
    @property
    def num_agents(self):
        return self.env.num_agents
    
    def __init__(self, env: MultiAgentEnv):
        """Initialize wrapper with base environment."""
        self.env = env
        
        # Store the number of actions for each agent
        self._num_actions = {}
        for i in range(env.num_agents):
            agent_id = f"agent_{i}"
            action_space = env.action_space(agent_id)
            if not isinstance(action_space, spaces.Discrete):
                raise ValueError(f"ActionObservableWrapper only supports Discrete action spaces. {agent_id} has {type(action_space)}")
            self._num_actions[agent_id] = action_space.n
        
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict, chex.Array]:
        """Reset environment and initialize action history."""
        obs, state = self.env.reset(key)
        
        # Initialize last_actions with zeros
        last_actions = jnp.zeros(self.env.num_agents, dtype=jnp.int32)
        
        # Add last_actions to each agent's observation using the same logic as step()
        def update_obs(i, obs):
            other_agents_actions = jnp.delete(last_actions, i)
            one_hot_actions = jax.nn.one_hot(other_agents_actions, self._num_actions)
            
            if len(obs[f"agent_{i}"].shape) == 3:
                action_layer = jnp.zeros((*obs[f"agent_{i}"].shape[:2], self._num_actions)
                action_layer = action_layer.at[0, 0].set(one_hot_actions.flatten())
                new_obs = jnp.concatenate([obs[f"agent_{i}"], action_layer], axis=-1)
            else:
                new_obs = jnp.concatenate([obs[f"agent_{i}"], one_hot_actions.flatten()])
            
            return obs.at[f"agent_{i}"].set(new_obs)
            
        obs = jax.lax.fori_loop(0, self.env.num_agents, update_obs, obs)
        
        # Store last_actions in state for next step
        state = state.replace(last_actions=last_actions)
        return obs, state
        
    def step(self, key: chex.PRNGKey, state: chex.Array, actions: Dict) -> Tuple[Dict, chex.Array, Dict, Dict, Dict]:
        """Execute environment step and augment observations with other agents' actions."""
        # Step the base environment
        obs, next_state, reward, done, info = self.env.step(key, state, actions)
        
        # Convert actions dictionary to array
        # e.g., {"agent_0": 1, "agent_1": 3} -> [1, 3]
        action_array = jnp.array([actions[f"agent_{i}"] for i in range(self.env.num_agents)])
        
        # Handle observations in a JAX-friendly way for each agent
        def update_obs(i, obs):
            agent_id = f"agent_{i}"
            # Get all actions except the current agent's action
            other_agents_actions = jnp.delete(action_array, i)
            other_agent_ids = [f"agent_{j}" for j in range(self.env.num_agents) if j != i]
            
            # Create one-hot encodings with appropriate sizes for each agent
            one_hot_actions = [
                jax.nn.one_hot(action, self._num_actions[other_id])
                for action, other_id in zip(other_agents_actions, other_agent_ids)
            ]
            # Combine all one-hot vectors into a single array
            flattened_actions = jnp.concatenate([oh.flatten() for oh in one_hot_actions])
            
            # Add to observation based on observation type
            if len(obs[agent_id].shape) == 3:  # Image observations
                action_layer = jnp.zeros((*obs[agent_id].shape[:2], flattened_actions.size))
                action_layer = action_layer.at[0, 0].set(flattened_actions)
                new_obs = jnp.concatenate([obs[agent_id], action_layer], axis=-1)
            else:  # Vector observations
                new_obs = jnp.concatenate([obs[agent_id], flattened_actions])
            
            return obs.at[agent_id].set(new_obs)
        
        obs = jax.lax.fori_loop(0, self.env.num_agents, update_obs, obs)
        
        # Store current actions for next step
        next_state = next_state.replace(last_actions=action_array)
        return obs, next_state, reward, done, info

    def action_space(self, agent: str) -> spaces.Box:
        """Return the action space for the specified agent."""
        return self.env.action_space(agent)

    def observation_space(self, agent: str) -> spaces.Box:
        """Define the new observation space that includes other agents' actions."""
        base_space = self.env.observation_space(agent)
        
        # Calculate total size of other agents' action spaces
        agent_idx = int(agent.split('_')[1])
        other_agents = [f"agent_{i}" for i in range(self.env.num_agents) if i != agent_idx]
        total_action_dims = sum(self._num_actions[other_id] for other_id in other_agents)
        
        if len(base_space.shape) == 3:  # Image observations
            # Add action channels to existing channels
            new_shape = (*base_space.shape[:2], base_space.shape[2] + total_action_dims)
        else:  # Vector observations
            # Add space for other agents' one-hot actions
            new_shape = (base_space.shape[0] + total_action_dims,)
            
        return spaces.Box(
            low=jnp.min(base_space.low),
            high=jnp.max(base_space.high),
            shape=new_shape,
            dtype=base_space.dtype
        ) 