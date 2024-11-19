# File: jaxmarl/wrappers/wandb_logger.py

import logging
from typing import Dict, Tuple, Any
import jax
import jax.numpy as jnp
import wandb
import chex
from flax import struct
from jaxmarl.environments import MultiAgentEnv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        debug: bool = False,
    ):
        super().__init__(num_agents=env.num_agents)
        self._env = env
        self.agents = env.agents
        self.debug = debug

        # Add action names for better visualization
        self.action_names = {
            0: "Up",
            1: "Down",
            2: "Right",
            3: "Left",
            4: "Stay",
            5: "Interact"
        }
        
        # Initialize metric storage
        self._episode_rewards = {agent: 0.0 for agent in self.agents}
        self._episode_lengths = {agent: 0 for agent in self.agents}
        self._episode_dishes = 0
        self._total_dishes = 0
        self._episode_collisions = 0
        self._proximity_count = 0  # Track time agents spend near each other
        self._joint_action_count = 0  # Track coordinated actions
        self._action_counts = jnp.zeros(6, dtype=jnp.int32) # 6 possible actions
        self._step_count = 0

        # Performance tracking
        self._best_return = float('-inf')
        self._best_dishes = 0
        
        # Action ratios
        self._movement_ratio = 0
        self._interaction_ratio = 0
        
        # Reward components
        self._onion_rewards = 0
        self._plate_rewards = 0
        
        # Initialize WandB run
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=experiment_name,
            tags=tags or [],
            group=group,
            config=config,
            reinit=True
        )
        
        # Set up custom WandB charts
        self._setup_wandb_charts()

    def _setup_wandb_charts(self):
        """Configure custom WandB charts and panels for comprehensive monitoring"""
        # Core metrics with appropriate summary statistics
        wandb.define_metric("episode/returns", summary="mean")
        wandb.define_metric("episode/lengths", summary="mean")
        wandb.define_metric("episode/completed_dishes", summary="max")
        wandb.define_metric("episode/agent_collisions", summary="mean")
        wandb.define_metric("episode/total_dishes", summary="max")
        
        # Action-specific metrics
        wandb.define_metric("actions/movement_ratio", summary="mean")
        wandb.define_metric("actions/interaction_ratio", summary="mean")
        
        # Training metrics
        wandb.define_metric("train/policy_loss", summary="mean")
        wandb.define_metric("train/value_loss", summary="mean")
        wandb.define_metric("train/entropy", summary="mean")
        
        # Create step metric for x-axis
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")
        
        # Initialize custom visualization tables with better descriptions
        self.run.summary.update({
            # Performance tracking
            "charts/performance": wandb.Table(
                columns=["Training Step", "Episode Returns", "Completed Dishes", "Episode Length"],
                data=[]
            ),
            
            # Coordination metrics
            "charts/coordination": wandb.Table(
                columns=["Training Step", "Agent Collisions", "Joint Actions", "Proximity Score"],
                data=[]
            ),
            
            # Action distribution
            "charts/action_dist": wandb.Table(
                columns=["Action Type", "Count", "Percentage"],
                data=[]
            ),
            
            # Task completion timeline
            "charts/task_completion": wandb.Table(
                columns=["Training Step", "Onions Collected", "Soups Prepared", "Dishes Delivered"],
                data=[]
            )
        })
        
        # Set up custom plots
        wandb.log({
            "charts/action_histogram": wandb.plot.histogram(
                wandb.Table(columns=["Action Type"]),
                "Action Type",
                title="Distribution of Agent Actions"
            ),
            
            "charts/training_progress": wandb.plot.line_series(
                xs=[0],
                ys=[[0], [0]],
                keys=["Episode Returns", "Completed Dishes"],
                title="Training Progress Over Time",
                xname="Training Steps"
            )
        })
    
    def _get_unwrapped_state(self, state):
        """Helper to unwrap state if needed."""
        original_type = type(state).__name__
        current_state = state
        unwrap_chain = [original_type]
        
        while hasattr(current_state, 'env_state'):
            current_state = current_state.env_state
            unwrap_chain.append(type(current_state).__name__)
        
        if len(unwrap_chain) > 1:
            print(f"State unwrapping chain: {' -> '.join(unwrap_chain)}")
        
        return current_state

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], Any]:
        """Reset environment and metrics"""
        obs, state = self._env.reset(key)
        
        # Store state type for consistent wrapping
        self._last_state_type = state
        
        # Reset episode metrics
        self._episode_rewards = {agent: 0.0 for agent in self.agents}
        self._episode_lengths = {agent: 0 for agent in self.agents}
        self._episode_dishes = 0
        self._episode_collisions = 0
        self._proximity_count = 0
        self._joint_action_count = 0
        self._action_counts = jnp.zeros(6, dtype=jnp.int32)
        self._onion_rewards = 0
        self._plate_rewards = 0
        
        return obs, state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: Any,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], Any, Dict[str, float], Dict[str, bool], Dict]:
        """Step environment and log metrics"""
        # Get unwrapped state for the base environment
        unwrapped_state = self._get_unwrapped_state(state)
        
        # Step the base environment
        obs, next_state, rewards, dones, info = self._env.step_env(key, unwrapped_state, actions)
        
        # Update step count and basic metrics
        self._step_count += 1
        for agent in self.agents:
            self._episode_rewards[agent] += rewards[agent]
            self._episode_lengths[agent] += 1
        
        # Convert actions to array for vectorized operations
        action_array = jnp.array([actions[agent] for agent in self.agents])
        
        # Track action statistics using JAX operations
        for idx in range(len(action_array)):
            self._action_counts = self._action_counts.at[action_array[idx]].add(1)
            
        # Calculate movement ratio (actions 0-3 are movement)
        movement_mask = jnp.where(action_array < 4, 1, 0)
        self._movement_ratio = jnp.sum(movement_mask) / len(action_array)
        
        # Calculate interaction ratio (action 5 is interact)
        interact_mask = jnp.where(action_array == 5, 1, 0)
        self._interaction_ratio = jnp.sum(interact_mask) / len(action_array)
        
        # Process rewards and track task completion
        shaped_rewards = info.get("shaped_reward", {})
        if shaped_rewards:
            self._onion_rewards += shaped_rewards.get("onion_rewards", 0)
            self._plate_rewards += shaped_rewards.get("plate_rewards", 0)
            if shaped_rewards.get("delivery_rewards", 0) > 0:
                self._episode_dishes += 1
                self._total_dishes += 1
        
        # Track best performance
        episode_return = jnp.sum(jnp.array(list(self._episode_rewards.values())))
        self._best_return = jnp.maximum(self._best_return, episode_return)
        self._best_dishes = jnp.maximum(self._best_dishes, self._total_dishes)
        
        # Log metrics
        if dones["__all__"]:
            self._log_episode_metrics()
        
        # Log step metrics
        metrics = {
            "step/mean_reward": jnp.mean(jnp.array(list(rewards.values()))),
            "step/movement_ratio": self._movement_ratio,
            "step/interaction_ratio": self._interaction_ratio,
            "step/proximity": self._proximity_count / self._step_count,
            "step": self._step_count,
        }
        
        if shaped_rewards:
            metrics.update({
                f"step/shaped_rewards/{k}": v 
                for k, v in shaped_rewards.items()
            })
        
        wandb.log(metrics)
        
        return obs, next_state, rewards, dones, info

    def _update_metrics(self, actions, rewards, next_state, info):
        """Update metrics with error checking."""
        try:
            # Update step count and basic metrics
            self._step_count += 1
            for agent in self.agents:
                self._episode_rewards[agent] += rewards[agent]
                self._episode_lengths[agent] += 1
            
            # Track action statistics
            for agent, action in actions.items():
                try:
                    self._action_counts = self._action_counts.at[int(action)].add(1)
                except Exception as e:
                    logger.error(f"Error updating action counts for agent {agent}: {e}")
            
            # Calculate ratios
            try:
                movement_actions = jnp.sum(jnp.where(jnp.array(list(actions.values())) < 4, 1, 0))
                self._movement_ratio = movement_actions / len(actions)
                interact_actions = jnp.sum(jnp.where(jnp.array(list(actions.values())) == 5, 1, 0))
                self._interaction_ratio = interact_actions / len(actions)
            except Exception as e:
                logger.error(f"Error calculating action ratios: {e}")
            
            # Update other metrics
            if self.debug:
                logger.debug(f"Current episode rewards: {self._episode_rewards}")
                logger.debug(f"Current action counts: {self._action_counts}")
                
        except Exception as e:
            logger.error(f"Error in _update_metrics: {e}")
            raise

    def _update_tables(self, step: int, metrics: dict):
        """Update the custom tables with new data"""
        # Update performance table
        self.run.summary["charts/performance"].add_data(
            step,
            metrics.get("episode/returns", 0.0),
            metrics.get("episode/completed_dishes", 0),
            metrics.get("episode/lengths", 0)
        )
        
        # Update coordination table
        self.run.summary["charts/coordination"].add_data(
            step,
            metrics.get("episode/agent_collisions", 0),
            metrics.get("episode/joint_actions", 0),
            metrics.get("episode/proximity", 0)
        )
        
        # Update action distribution
        action_counts = metrics.get("episode/action_distribution", jnp.zeros(6))
        total_actions = action_counts.sum()
        for action, count in enumerate(action_counts):
            percentage = (count / total_actions) * 100 if total_actions > 0 else 0
            self.run.summary["charts/action_dist"].add_data(
                self.action_names[action],  # Convert index to action name
                float(count),
                float(percentage)
            )

    def _log_episode_metrics(self):
        """Enhanced metric logging with visual updates"""
        metrics = {
            "episode/returns": jnp.mean(jnp.array(list(self._episode_rewards.values()))),
            "episode/lengths": jnp.mean(jnp.array(list(self._episode_lengths.values()))),
            "episode/completed_dishes": self._episode_dishes,
            "episode/total_dishes": self._total_dishes,
            "episode/agent_collisions": self._episode_collisions,
            "episode/action_distribution": self._action_counts,
            "episode/onion_rewards": self._onion_rewards,
            "episode/plate_rewards": self._plate_rewards,
            "episode/best_return": self._best_return,
            "episode/best_dishes": self._best_dishes,
            "step": self._step_count
        }
        
        # Log metrics
        wandb.log(metrics)
        
        # Update custom tables and visualizations
        self._update_tables(self._step_count, metrics)
        
        # Update histograms and plots
        wandb.log({
            "charts/action_histogram": wandb.plot.histogram(
                wandb.Table(columns=["action"], 
                        data=[[a] for a in self._action_counts]),
                "action"
            )
        })

    def _log_step_metrics(self, actions, rewards, info):
        """Log metrics at each step"""
        metrics = {
            "step/mean_reward": jnp.mean(jnp.array(list(rewards.values()))),
            "step/movement_ratio": self._movement_ratio,
            "step/interaction_ratio": self._interaction_ratio,
            "step/proximity": self._proximity_count / self._step_count,
            "step": self._step_count,
        }
        
        # Add shaped rewards to metrics if available
        if "shaped_reward" in info:
            shaped_rewards = info["shaped_reward"]
            metrics.update({
                f"step/shaped_rewards/{k}": v 
                for k, v in shaped_rewards.items()
            })
        
        wandb.log(metrics)

    def close(self):
        """Cleanup with error handling."""
        try:
            if self.debug:
                logger.info("Closing WandbMonitorWrapper")
            wandb.finish()
        except Exception as e:
            logger.error(f"Error closing wandb: {e}")

    # Forward other methods to wrapped env
    def observation_space(self):
        return self._env.observation_space()

    def action_space(self, agent_id: str = ""):
        return self._env.action_space(agent_id)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _safe_update_metric(self, metric_name: str, value: float):
        """Safely update a metric value with error handling.
        
        Args:
            metric_name: Name of the metric to update
            value: New value for the metric
        """
        try:
            float_value = float(value)
            wandb.log({metric_name: float_value})
            if self.debug:
                logger.debug(f"Updated metric {metric_name}: {float_value}")
        except ValueError as e:
            logger.error(f"Error converting metric value to float: {e}")
        except Exception as e:
            logger.error(f"Error logging metric {metric_name}: {e}")