from typing import Dict, Tuple, Any
import jax
import jax.numpy as jnp
import chex
from flax import struct
from jaxmarl.environments import MultiAgentEnv
from jaxmarl.environments import spaces
from jaxmarl.environments.overcooked.overcooked import Actions

class ActionAwareWrapper(MultiAgentEnv):
    """A wrapper that enhances the environment by adding co-player's previous actions 
    to each agent's observations. This helps agents learn to coordinate by being aware of their 
    partner's recent actions.
    
    The wrapper adds one additional channel to the observation space, making the
    total number of channels equivalent to the original + 1 for co-player's action.
    """
    
    def __init__(self, env: MultiAgentEnv):
        # Initialize parent class with same number of agents as base environment
        super().__init__(num_agents=env.num_agents)
        # Store reference to base environment
        self._env = env
        # Store list of agents
        self.agents = env.agents
        
        # Get base observation shape and space
        self._base_obs_space = env.observation_space()
        # Store shape of base observation space
        self._base_obs_shape = self._base_obs_space.shape
        print("Base observation shape:", self._base_obs_shape)  # Debug
        
        # Extend observation shape to include action channel
        self.obs_shape = (
            self._base_obs_shape[0], # Width
            self._base_obs_shape[1], # Height
            self._base_obs_shape[2] + 1  # Add an additional channel for action
        )
        print("Wrapped observation shape:", self.obs_shape)  # Debug
        
        # Initialize storage for previous actions (will be set in reset)
        self._prev_actions = None

    def _get_unwrapped_state(self, state):
        """Recursively unwrap state object until we get to the base state.
        
        This is needed because the environment might be wrapped multiple times
        (e.g., by LogWrapper), and we need to access the original state object.
        
        Args:
            state: The possibly wrapped state object
            
        Returns:
            The unwrapped base state object
        """
        while hasattr(state, 'env_state'):
            state = state.env_state
        return state

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], Any]:
        """Reset environment and initialize previous actions"""
        obs, state = self._env.reset(key)
        
        # Initialize previous actions as "stay"
        self._prev_actions = {
            agent: jnp.full((1,), Actions.stay, dtype=jnp.int32)
            for agent in self.agents
        }
        
        # Augment observations with previous actions
        obs = self._augment_obs(obs)
        
        # Unwrap state if needed
        state = self._get_unwrapped_state(state)
        
        return obs, state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: Any,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], Any, Dict[str, float], Dict[str, bool], Dict]:
        """Step environment and update previous actions"""
        # Unwrap state if needed
        unwrapped_state = self._get_unwrapped_state(state)
        
        # Step the base environment
        obs, state, rewards, dones, info = self._env.step_env(key, unwrapped_state, actions)
        
        # Update previous actions
        self._prev_actions = actions
        
        # Augment observations with previous actions
        obs = self._augment_obs(obs)
        
        return obs, state, rewards, dones, info

    def _augment_obs(self, obs: Dict[str, chex.Array]) -> Dict[str, chex.Array]:
        """Add co-player's previous action to each agent's observation"""
        augmented_obs = {}
        for i, agent in enumerate(self.agents):
            # Get co-player
            coplayer = self.agents[1 - i]
            
            # Get co-player's previous action and ensure it's the right shape
            coplayer_action = self._prev_actions[coplayer]
            
            # Create action channel with correct shape
            # Remove extra dimensions from action and broadcast to match spatial dims
            action_channel = jnp.broadcast_to(
                coplayer_action.reshape(-1, 1),  # Reshape to (batch_size, 1)
                (*obs[agent].shape[:-1], 1)  # Match spatial dims with single channel
            )
            
            # Concatenate base observation with action channel
            augmented_obs[agent] = jnp.concatenate(
                [obs[agent], action_channel],
                axis=-1
            )
            
        return augmented_obs

    def observation_space(self) -> spaces.Box:
        """Updated observation space to include action channel"""
        base_space = self._env.observation_space()
        
        # Handle both integer and array low/high values
        if isinstance(base_space.low, (int, float)):
            low = base_space.low
            high = base_space.high
        else:
            low = base_space.low.min()
            high = base_space.high.max()
        
        return spaces.Box(
            low=low,
            high=high,
            shape=self.obs_shape,
            dtype=base_space.dtype
        )

    # Delegate other methods to base environment
    def action_space(self, agent_id: str = "") -> spaces.Discrete:
        return self._env.action_space(agent_id)

    @property
    def name(self) -> str:
        return f"ActionAware{self._env.name}"

    # Forward any other environment properties/methods to base env
    def __getattr__(self, name):
        return getattr(self._env, name)