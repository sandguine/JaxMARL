# File: jaxmarl/wrappers/wandb_logger.py

from typing import Dict, Tuple, Any
import jax
import jax.numpy as jnp
import wandb
import chex
from flax import struct
from jaxmarl.environments import MultiAgentEnv

class WandbMonitorWrapper(MultiAgentEnv):
    """Wrapper for logging Overcooked metrics to WandB.
    Handles metric collection, processing and visualization.
    """
    
    def __init__(
        self, 
        env: MultiAgentEnv,
        experiment_name: str,
        project: str,
        entity: str,
        tags: list = None,
        group: str = None,
        config: dict = None,
    ):
        super().__init__(num_agents=env.num_agents)
        self._env = env
        self.agents = env.agents
        
        # Initialize WandB run
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=experiment_name,
            tags=tags,
            group=group,
            config=config,
        )
        
        # Initialize metric storage
        self._episode_rewards = {agent: 0.0 for agent in self.agents}
        self._episode_lengths = {agent: 0 for agent in self.agents}
        self._episode_dishes = 0
        self._episode_collisions = 0
        self._action_counts = jnp.zeros(6)  # 6 possible actions
        self._step_count = 0
        
        # Set up custom WandB charts
        self._setup_wandb_charts()

    def _setup_wandb_charts(self):
        """Configure custom WandB charts and panels"""
        # Define main metrics
        wandb.define_metric("episode_returns", summary="mean")
        wandb.define_metric("episode_lengths", summary="mean")
        wandb.define_metric("completed_dishes", summary="sum")
        
        # Create custom x-axis
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")
        
        # Create custom panels
        self.run.summary.update({
            "charts/performance": wandb.Table(
                columns=["step", "returns", "dishes", "length"]
            ),
            "charts/coordination": wandb.Table(
                columns=["step", "collisions", "joint_actions"]
            ),
            "charts/action_dist": wandb.Table(
                columns=["action_type", "count"]
            )
        })

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], Any]:
        """Reset environment and metrics"""
        obs, state = self._env.reset(key)
        
        # Reset episode metrics
        self._episode_rewards = {agent: 0.0 for agent in self.agents}
        self._episode_lengths = {agent: 0 for agent in self.agents}
        self._episode_dishes = 0
        self._episode_collisions = 0
        self._action_counts = jnp.zeros(6)
        
        return obs, state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: Any,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], Any, Dict[str, float], Dict[str, bool], Dict]:
        """Step environment and log metrics"""
        obs, next_state, rewards, dones, info = self._env.step_env(key, state, actions)
        
        # Update metrics
        self._step_count += 1
        for agent in self.agents:
            self._episode_rewards[agent] += rewards[agent]
            self._episode_lengths[agent] += 1
        
        # Update action counts
        for agent, action in actions.items():
            self._action_counts = self._action_counts.at[action].add(1)
        
        # Check for collisions (simplified)
        pos_a = next_state.agent_pos[0]
        pos_b = next_state.agent_pos[1]
        collision = jnp.all(pos_a == pos_b)
        self._episode_collisions += collision
        
        # Detect completed dishes (from info)
        self._episode_dishes += info.get("completed_dishes", 0)
        
        # Log metrics if episode is done
        if dones["__all__"]:
            self._log_episode_metrics()
        
        # Log step metrics
        self._log_step_metrics(actions, rewards, info)
        
        return obs, next_state, rewards, dones, info

    def _log_episode_metrics(self):
        """Log metrics at episode end"""
        metrics = {
            "episode/returns": jnp.mean(jnp.array(list(self._episode_rewards.values()))),
            "episode/length": jnp.mean(jnp.array(list(self._episode_lengths.values()))),
            "episode/completed_dishes": self._episode_dishes,
            "episode/collisions": self._episode_collisions,
            "episode/action_distribution": self._action_counts / self._action_counts.sum(),
        }
        wandb.log(metrics, step=self._step_count)

    def _log_step_metrics(self, actions, rewards, info):
        """Log metrics at each step"""
        metrics = {
            "step/mean_reward": jnp.mean(jnp.array(list(rewards.values()))),
            "step/shaped_rewards/onion_rewards": info.get("onion_rewards", 0),
            "step/shaped_rewards/plate_rewards": info.get("plate_rewards", 0),
            "step/shaped_rewards/delivery_rewards": info.get("delivery_rewards", 0),
            "step": self._step_count,
        }
        wandb.log(metrics, step=self._step_count)

    def close(self):
        """Finish WandB run"""
        wandb.finish()

    # Forward other methods to wrapped env
    def observation_space(self):
        return self._env.observation_space()

    def action_space(self, agent_id: str = ""):
        return self._env.action_space(agent_id)

    def __getattr__(self, name):
        return getattr(self._env, name)