# File: jaxmarl/wrappers/wandb_logger.py

import logging
from typing import Dict, Tuple, Any
import jax
import jax.numpy as jnp
import wandb
import chex
from flax import struct
from jaxmarl.environments import MultiAgentEnv
import datetime
import numpy as np

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
        
        # Initialize metric storage with explicit Python types
        self._episode_rewards = {agent: 0.0 for agent in self.agents}
        self._episode_lengths = {agent: 0 for agent in self.agents}
        self._episode_dishes = 0
        self._total_dishes = 0
        self._episode_collisions = 0
        self._proximity_count = 0 # Track time agents spend near each other
        self._joint_action_count = 0 # Track coordinated actions
        self._action_counts = jnp.zeros(6, dtype=jnp.int32) # 6 possible actions
        self._step_count = 0

        # Performance tracking
        self._best_return = float('-inf')
        self._best_dishes = 0
        
        # Action ratios
        self._movement_ratio = 0.0
        self._interaction_ratio = 0.0
        
        # Reward components
        self._onion_rewards = 0.0
        self._plate_rewards = 0.0
        
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
        """Configure custom WandB charts and panels with comprehensive monitoring"""
        try:
            # Define episode metrics with appropriate summaries
            metrics_config = {
                "episode/returns": "mean",
                "episode/lengths": "mean",
                "episode/completed_dishes": "sum",
                "episode/agent_collisions": "mean",
                "episode/total_dishes": "max",
                "episode/movement_ratio": "mean",
                "episode/interaction_ratio": "mean",
                "train/policy_loss": "mean",
                "train/value_loss": "mean",
                "train/entropy": "mean"
            }
            
            # Configure all metrics
            for metric_name, summary_type in metrics_config.items():
                wandb.define_metric(metric_name, summary=summary_type)
                
            # Define step as the global x-axis
            wandb.define_metric("step")
            wandb.define_metric("*", step_metric="step")
            
            # Initialize tables with clear structure
            tables_config = {
                "performance_metrics": {
                    "columns": ["step", "returns", "completed_dishes", "episode_length"],
                    "description": "Core performance metrics over time"
                },
                "action_distribution": {
                    "columns": ["step", "action_type", "count", "percentage"],
                    "description": "Distribution of agent actions"
                },
                "coordination_metrics": {
                    "columns": ["step", "collisions", "joint_actions", "proximity"],
                    "description": "Agent coordination indicators"
                }
            }
            
            # Create and log tables
            self.tables = {}
            for table_name, config in tables_config.items():
                try:
                    table = wandb.Table(
                        columns=config["columns"],
                        data=[],  # Start empty
                    )
                    self.tables[table_name] = table
                    wandb.log({f"tables/{table_name}": table})
                    
                    if self.debug:
                        logger.debug(f"Created table: {table_name}")
                        
                except Exception as e:
                    logger.error(f"Failed to create table {table_name}: {e}")
                    
            # Initialize visualization plots
            try:
                initial_plots = {
                    "training/progress": wandb.plot.line_series(
                        xs=[[0]],
                        ys=[[0]],
                        keys=["Episode Returns"],
                        title="Training Progress",
                        xname="Training Steps"
                    ),
                    
                    "training/dishes_completed": wandb.plot.line_series(
                        xs=[[0]],
                        ys=[[0]],
                        keys=["Completed Dishes"],
                        title="Dishes Completed Over Time",
                        xname="Training Steps"
                    ),
                    
                    "actions/distribution": wandb.plot.bar(
                        table=self.tables["action_distribution"],
                        value="count",
                        label="action_type",
                        title="Action Distribution"
                    )
                }
                
                wandb.log(initial_plots)
                
                if self.debug:
                    logger.debug("Successfully initialized visualization plots")
                    
            except Exception as e:
                logger.error(f"Failed to create initial plots: {e}")
                
            # Store table references for updates
            self.run.summary.update({
                f"table_{name}": table 
                for name, table in self.tables.items()
            })
            
            if self.debug:
                logger.debug("WandB charts and tables setup completed successfully")
                
        except Exception as e:
            logger.error(f"Critical error in _setup_wandb_charts: {str(e)}")
            raise

    def _get_unwrapped_state(self, state: Any) -> Any:
        """Helper to safely unwrap nested environment states with comprehensive error handling.
        
        Args:
            state: The current environment state, potentially wrapped
        
        Returns:
            The unwrapped base environment state
        
        Note:
            This method handles nested environment wrappers and provides detailed
            debugging information about the unwrapping chain.
        """
        try:
            # Store original state type for error handling
            original_type = type(state).__name__
            
            # Initialize unwrapping tracking
            unwrap_chain = [original_type]
            current_state = state
            unwrap_depth = 0
            max_unwrap_depth = 10  # Prevent infinite loops
            
            # Attempt to unwrap state
            while hasattr(current_state, 'env_state'):
                # Safety check for maximum unwrap depth
                unwrap_depth += 1
                if unwrap_depth > max_unwrap_depth:
                    logger.warning(f"Maximum unwrap depth ({max_unwrap_depth}) exceeded. Possible circular reference.")
                    break
                    
                try:
                    # Track the current wrapper type
                    current_type = type(current_state).__name__
                    unwrap_chain.append(current_type)
                    
                    # Unwrap one level
                    current_state = current_state.env_state
                    
                    # Validate unwrapped state
                    if current_state is None:
                        logger.error("Encountered None state during unwrapping")
                        break
                    
                except AttributeError as ae:
                    logger.error(f"AttributeError while unwrapping state at depth {unwrap_depth}: {ae}")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error while unwrapping state at depth {unwrap_depth}: {e}")
                    break
            
            # Add final unwrapped state type to chain
            unwrap_chain.append(type(current_state).__name__)
            
            # Log unwrapping information if debug is enabled
            if self.debug:
                if len(unwrap_chain) > 1:
                    logger.debug(f"State unwrapping chain: {' -> '.join(unwrap_chain)}")
                    logger.debug(f"Unwrap depth: {unwrap_depth}")
                else:
                    logger.debug("State did not require unwrapping")
                    
            # Validate final state
            if current_state is None:
                logger.error("Unwrapping resulted in None state, returning original")
                return state
            
            return current_state
            
        except Exception as e:
            logger.error(f"Critical error in _get_unwrapped_state: {e}")
            logger.error(f"Original state type: {original_type}")
            logger.error(f"Unwrap chain: {' -> '.join(unwrap_chain)}")
            # Return original state as fallback
            return state
            
        finally:
            # Log memory usage if debug is enabled
            if self.debug:
                try:
                    import psutil
                    process = psutil.Process()
                    logger.debug(f"Memory usage after state unwrap: {process.memory_info().rss / 1024 / 1024:.2f} MB")
                except ImportError:
                    pass

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], Any]:
        """Reset environment and metrics"""
        obs, state = self._env.reset(key)
        
        # Reset episode metrics
        self._episode_rewards = {agent: 0.0 for agent in self.agents}
        self._episode_lengths = {agent: 0 for agent in self.agents}
        self._episode_dishes = 0
        self._episode_collisions = 0
        self._proximity_count = 0
        self._joint_action_count = 0
        self._action_counts = jnp.zeros(6, dtype=jnp.int32)
        self._onion_rewards = 0.0
        self._plate_rewards = 0.0
        
        return obs, state

    def step_env(
        self, 
        key: chex.PRNGKey, 
        state: Any, 
        actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], Any, Dict[str, float], Dict[str, bool], Dict]:
        """Execute environment step with comprehensive metric tracking and error handling.
        
        Args:
            key: PRNG key for randomization
            state: Current environment state
            actions: Dictionary of agent actions
            
        Returns:
            Tuple containing:
            - observations: Dict of agent observations
            - next_state: Next environment state
            - rewards: Dict of agent rewards
            - dones: Dict of done flags
            - info: Additional information dictionary
            
        Raises:
            Exception: If environment step fails
        """
        try:
            # Validate inputs
            if not isinstance(actions, dict):
                raise ValueError(f"Expected dict for actions, got {type(actions)}")
            
            # Get unwrapped state safely
            try:
                unwrapped_state = self._get_unwrapped_state(state)
            except Exception as e:
                logger.error(f"Error unwrapping state: {e}")
                unwrapped_state = state  # Fallback to original state
                
            # Execute environment step with error handling
            try:
                obs, next_state, rewards, dones, info = self._env.step_env(
                    key, unwrapped_state, actions
                )
            except Exception as e:
                logger.error(f"Environment step failed: {e}")
                raise
                
            # Convert rewards and dones to Python types
            try:
                rewards = {
                    k: float(v) if isinstance(v, (jax.numpy.ndarray, jax.core.Tracer)) else v 
                    for k, v in rewards.items()
                }
                dones = {
                    k: bool(v) if isinstance(v, (jax.numpy.ndarray, jax.core.Tracer)) else v 
                    for k, v in dones.items()
                }
            except Exception as e:
                logger.error(f"Error converting rewards/dones: {e}")
                
            # Track metrics for this step
            try:
                metrics = {
                    "step/mean_reward": float(np.mean(list(rewards.values()))),
                    "step/total_reward": float(np.sum(list(rewards.values()))),
                    "step": self._step_count,
                }
                
                # Track action distribution
                action_counts = np.zeros(self._env.action_space().n)
                for action in actions.values():
                    action_counts[int(action)] += 1
                metrics["step/action_distribution"] = action_counts.tolist()
                
                # Add any custom metrics from info dict
                if isinstance(info, dict):
                    for k, v in info.items():
                        if isinstance(v, (int, float, bool)):
                            metrics[f"step/info_{k}"] = v
                            
                # Track episode completion
                if "episode" in info:
                    episode_info = info["episode"]
                    if isinstance(episode_info, dict):
                        for k, v in episode_info.items():
                            metrics[f"episode/{k}"] = v
                            
            except Exception as e:
                logger.error(f"Error computing metrics: {e}")
                metrics = {"step": self._step_count}
                
            # Log metrics
            try:
                # Log episode metrics if done
                if dones.get("__all__", False):
                    self._log_episode_metrics()
                    
                # Log step metrics
                self._log_metrics(metrics, self._step_count)
                
            except Exception as e:
                logger.error(f"Error logging metrics: {e}")
                
            # Update internal state
            try:
                self._step_count += 1
                self._update_internal_state(actions, rewards, next_state, info)
            except Exception as e:
                logger.error(f"Error updating internal state: {e}")
                
            # Debug logging
            if self.debug:
                logger.debug(f"Step {self._step_count} completed")
                logger.debug(f"Mean reward: {metrics.get('step/mean_reward', 0.0)}")
                
            return obs, next_state, rewards, dones, info
            
        except Exception as e:
            logger.error(f"Critical error in step_env: {e}")
            logger.error(f"Actions: {actions}")
            logger.error(f"Step: {self._step_count}")
            raise
            
        finally:
            # Monitor memory usage in debug mode
            if self.debug:
                try:
                    import psutil
                    process = psutil.Process()
                    logger.debug(f"Memory usage after step: {process.memory_info().rss / 1024 / 1024:.2f} MB")
                except ImportError:
                    pass

    def _update_metrics(self, actions, rewards, next_state, info):
        """Update metrics with error checking."""
        try:
            # Update step count and basic metrics
            self._step_count += 1
            for agent in self.agents:
                # Convert rewards to float to avoid JAX tracer issues
                self._episode_rewards[agent] += float(rewards[agent])
                self._episode_lengths[agent] += 1
            
            # Track action statistics
            for agent, action in actions.items():
                try:
                    # Convert action to int before using as index
                    action_idx = int(action)
                    self._action_counts = self._action_counts.at[action_idx].add(1)
                except Exception as e:
                    logger.error(f"Error updating action counts for agent {agent}: {e}")
            
            # Calculate ratios with explicit conversion
            try:
                action_values = jnp.array(list(actions.values()))
                movement_actions = float(jnp.sum(jnp.where(action_values < 4, 1, 0)))
                self._movement_ratio = movement_actions / len(actions)
                interact_actions = float(jnp.sum(jnp.where(action_values == 5, 1, 0)))
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
        """Update the custom tables with new data.
        
        Args:
            step: Current training step
            metrics: Dictionary of metrics to update tables with
        """
        try:
            # Handle None metrics
            if metrics is None:
                logger.warning("Received None metrics in _update_tables")
                metrics = {}
            
            # First ensure all values are converted from JAX types
            safe_metrics = {}
            for k, v in metrics.items():
                try:
                    if isinstance(v, (jax.numpy.ndarray, jax.core.Tracer)):
                        if v.ndim == 0:
                            safe_metrics[k] = float(v.item())
                        else:
                            safe_metrics[k] = [float(x) if hasattr(x, 'item') else x for x in v]
                    elif isinstance(v, (list, tuple)):
                        safe_metrics[k] = [
                            float(x) if isinstance(x, (jax.numpy.ndarray, jax.core.Tracer)) else x 
                            for x in v
                        ]
                    else:
                        safe_metrics[k] = v
                except Exception as e:
                    logger.error(f"Error converting metric {k}: {e}")
                    safe_metrics[k] = 0.0  # Fallback value
                
            # Ensure required tables exist
            required_tables = [
                "performance_metrics",
                "coordination_metrics",
                "action_distribution",
                "task_completion"
            ]
            for table in required_tables:
                if table not in self.run.summary:
                    logger.warning(f"Table {table} not found, recreating...")
                    self._setup_wandb_charts()

            try:
                # Update performance tracking table
                self.run.summary["performance_metrics"].add_data(
                    int(step),
                    float(safe_metrics.get("episode/returns", 0.0)),
                    int(safe_metrics.get("episode/completed_dishes", 0)),
                    float(safe_metrics.get("episode/lengths", 0.0))
                )
            except Exception as e:
                logger.error(f"Error updating performance table: {e}")

            try:
                # Update coordination metrics table
                self.run.summary["coordination_metrics"].add_data(
                    int(step),
                    float(safe_metrics.get("episode/agent_collisions", 0.0)),
                    float(safe_metrics.get("episode/joint_actions", 0.0)),
                    float(safe_metrics.get("episode/proximity", 0.0))
                )
            except Exception as e:
                logger.error(f"Error updating coordination table: {e}")

            # Update action distribution table
            try:
                action_counts = safe_metrics.get("episode/action_distribution", [0] * 6)
                if isinstance(action_counts, (jax.numpy.ndarray, jax.core.Tracer)):
                    action_counts = [float(x) for x in action_counts]
                total_actions = sum(action_counts)

                # Clear existing action distribution data safely
                if hasattr(self.run.summary["action_distribution"], "data"):
                    self.run.summary["action_distribution"].data = []
                
                # Add new action distribution data
                for action_idx, count in enumerate(action_counts):
                    if action_idx in self.action_names:  # Check if action index is valid
                        percentage = (count / total_actions * 100) if total_actions > 0 else 0.0
                        self.run.summary["action_distribution"].add_data(
                            self.action_names[action_idx],  # Action name
                            float(count),                   # Count
                            float(percentage)               # Percentage
                        )
            except Exception as e:
                logger.error(f"Error updating action distribution table: {e}")

            try:
                # Update task completion timeline
                self.run.summary["task_completion"].add_data(
                    int(step),
                    float(safe_metrics.get("episode/onion_rewards", 0.0)),
                    float(safe_metrics.get("episode/plate_rewards", 0.0)),
                    float(safe_metrics.get("episode/completed_dishes", 0.0))
                )
            except Exception as e:
                logger.error(f"Error updating task completion table: {e}")

            # Update custom plots
            try:
                wandb.log({
                    "visualizations/action_histogram": wandb.plot.histogram(
                        self.run.summary["action_distribution"],
                        "action_type",
                        title="Distribution of Agent Actions"
                    ),
                    
                    "visualizations/training_progress": wandb.plot.line_series(
                        xs=[step],
                        ys=[[safe_metrics.get("episode/returns", 0.0)], 
                            [safe_metrics.get("episode/completed_dishes", 0)]],
                        keys=["Episode Returns", "Completed Dishes"],
                        title="Training Progress Over Time",
                        xname="Training Steps"
                    )
                })
            except Exception as e:
                logger.error(f"Error updating custom plots: {e}")

            if self.debug:
                logger.debug(f"Successfully updated tables with metrics: {safe_metrics}")

        except Exception as e:
            logger.error(f"Critical error in _update_tables: {e}")
            logger.error(f"Problematic metrics: {metrics}")
            logger.error(f"Converted metrics: {safe_metrics if 'safe_metrics' in locals() else 'Not reached'}")
            raise

    def _log_episode_metrics(self):
        """Enhanced metric logging with visual updates and comprehensive error handling."""
        try:
            # Safely compute means with error handling
            try:
                returns_array = jnp.array(list(self._episode_rewards.values()))
                lengths_array = jnp.array(list(self._episode_lengths.values()))
                returns_mean = float(jnp.mean(returns_array)) if len(returns_array) > 0 else 0.0
                lengths_mean = float(jnp.mean(lengths_array)) if len(lengths_array) > 0 else 0.0
            except Exception as e:
                logger.error(f"Error computing means: {e}")
                returns_mean = 0.0
                lengths_mean = 0.0

            # Gather all metrics with explicit type conversion
            metrics = {
                "episode/returns": returns_mean,
                "episode/lengths": lengths_mean,
                "episode/completed_dishes": int(self._episode_dishes),
                "episode/total_dishes": int(self._total_dishes),
                "episode/agent_collisions": int(self._episode_collisions),
                "episode/action_distribution": [int(x) for x in self._action_counts],
                "episode/onion_rewards": float(self._onion_rewards),
                "episode/plate_rewards": float(self._plate_rewards),
                "episode/best_return": float(self._best_return),
                "episode/best_dishes": int(self._best_dishes),
                "step": int(self._step_count)
            }
            
            # Calculate additional metrics safely
            try:
                action_counts = metrics["episode/action_distribution"]
                total_actions = sum(action_counts)
                if total_actions > 0:
                    metrics.update({
                        "episode/movement_ratio": float(self._movement_ratio),
                        "episode/interaction_ratio": float(self._interaction_ratio),
                    })
            except Exception as e:
                logger.error(f"Error calculating additional metrics: {e}")
                metrics.update({
                    "episode/movement_ratio": 0.0,
                    "episode/interaction_ratio": 0.0,
                })

            # Convert any remaining JAX values
            try:
                final_metrics = {}
                for key, value in metrics.items():
                    if isinstance(value, (jax.numpy.ndarray, jax.core.Tracer)):
                        if value.ndim == 0:
                            final_metrics[key] = float(value)
                        else:
                            final_metrics[key] = [float(x) for x in value]
                    elif isinstance(value, (list, tuple)):
                        final_metrics[key] = [
                            float(x) if isinstance(x, (jax.numpy.ndarray, jax.core.Tracer)) else x 
                            for x in value
                        ]
                    else:
                        final_metrics[key] = value
            except Exception as e:
                logger.error(f"Error in final metric conversion: {e}")
                final_metrics = metrics  # Fallback to original metrics

            # Log metrics to wandb
            try:
                wandb.log(final_metrics)
            except Exception as e:
                logger.error(f"Error logging to wandb: {e}")

            # Update custom tables and visualizations
            try:
                self._update_tables(self._step_count, final_metrics)
            except Exception as e:
                logger.error(f"Error updating tables: {e}")

            # Update action distribution visualization
            try:
                action_dist_data = [
                    [self.action_names.get(i, f"Action_{i}"), count] 
                    for i, count in enumerate(final_metrics.get("episode/action_distribution", [0] * 6))
                ]
                
                wandb.log({
                    "charts/action_histogram": wandb.plot.histogram(
                        wandb.Table(
                            columns=["action", "count"],
                            data=action_dist_data
                        ),
                        "action",
                        title="Action Distribution"
                    )
                })
            except Exception as e:
                logger.error(f"Error updating action histogram: {e}")

            # Debug logging if enabled
            if self.debug:
                logger.debug(f"Logged episode metrics: {final_metrics}")

            # Reset episode-specific metrics with explicit types
            try:
                self._episode_rewards = {agent: 0.0 for agent in self.agents}
                self._episode_lengths = {agent: 0 for agent in self.agents}
                self._episode_dishes = 0
                self._episode_collisions = 0
                self._proximity_count = 0
                self._joint_action_count = 0
                self._action_counts = jnp.zeros(6, dtype=jnp.int32)
                self._onion_rewards = 0.0
                self._plate_rewards = 0.0
            except Exception as e:
                logger.error(f"Error resetting metrics: {e}")

        except Exception as e:
            logger.error("Critical error in _log_episode_metrics")
            logger.error(f"Error details: {str(e)}")
            logger.error(f"Current metrics state:")
            logger.error(f"- Episode rewards: {self._episode_rewards}")
            logger.error(f"- Episode lengths: {self._episode_lengths}")
            logger.error(f"- Action counts: {self._action_counts}")
            raise

    def _log_step_metrics(self, actions, rewards, info):
        """Log metrics at each step"""
        try:
            metrics = {
                "step/mean_reward": float(jnp.mean(jnp.array(list(rewards.values())))),
                "step/movement_ratio": float(self._movement_ratio),
                "step/interaction_ratio": float(self._interaction_ratio),
                "step/proximity": float(self._proximity_count) / float(self._step_count),
                "step": int(self._step_count),
            }
            
            # Add shaped rewards to metrics if available
            if "shaped_reward" in info:
                shaped_rewards = info["shaped_reward"]
                metrics.update({
                    f"step/shaped_rewards/{k}": float(v) 
                    for k, v in shaped_rewards.items()
                })
            
            wandb.log(metrics)
        except Exception as e:
            logger.error(f"Error logging step metrics: {e}")

    def close(self):
        """Cleanup with error handling."""
        try:
            if hasattr(self, 'run') and self.run is not None:
                if self.debug:
                    logger.info("Closing WandbMonitorWrapper")
                self.run.finish()
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
            # Convert JAX values to Python float
            if isinstance(value, (jax.numpy.ndarray, jax.core.Tracer)):
                float_value = float(value)
            else:
                float_value = float(value)
            wandb.log({metric_name: float_value})
            if self.debug:
                logger.debug(f"Updated metric {metric_name}: {float_value}")
        except Exception as e:
            logger.error(f"Error logging metric {metric_name}: {e}")

    def _convert_jax_value(self, value):
        """Helper to safely convert JAX values to Python types"""
        try:
            if isinstance(value, (jax.numpy.ndarray, jax.core.Tracer)):
                if value.ndim == 0:  # scalar
                    return value.item()
                return jnp.array(value)  # convert to regular array
            return value
        except Exception as e:
            logger.error(f"Error converting JAX value: {e}")
            return value

    def _convert_batch(self, batch_data: Any) -> Any:
        """Convert batch of JAX values to Python types with comprehensive error handling.
        
        Args:
            batch_data: Input data of any type that might contain JAX values
            
        Returns:
            Converted data with all JAX values converted to Python native types
        """
        try:
            # Handle None values
            if batch_data is None:
                return None
            
            # Handle dictionaries recursively
            if isinstance(batch_data, dict):
                return {
                    k: self._convert_batch(v) 
                    for k, v in batch_data.items()
                }
            
            # Handle lists and tuples while preserving type
            if isinstance(batch_data, (list, tuple)):
                converted = [self._convert_batch(x) for x in batch_data]
                return type(batch_data)(converted)
            
            # Handle JAX arrays and tracers
            if isinstance(batch_data, (jax.numpy.ndarray, jax.core.Tracer)):
                try:
                    # Move data to CPU if needed
                    data = jax.device_get(batch_data)
                    
                    # Handle scalar values
                    if data.ndim == 0:
                        return float(data.item())
                    
                    # Handle arrays
                    if data.shape:
                        # Convert to list for better serialization
                        return [float(x) for x in data.flatten()]
                    else:
                        return float(data.item())
                    
                except Exception as e:
                    logger.error(f"Error converting JAX array: {e}")
                    if self.debug:
                        logger.debug(f"Problematic data: {batch_data}")
                    return 0.0
                
            # Return other types unchanged
            return batch_data
            
        except Exception as e:
            logger.error(f"Critical error in _convert_batch: {e}")
            if self.debug:
                logger.debug(f"Failed to convert: {batch_data}")
            return batch_data

    def _log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Safely log metrics to WandB with comprehensive error handling and validation.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current training step
        """
        try:
            # Validate inputs
            if not isinstance(metrics, dict):
                raise ValueError(f"Expected dict for metrics, got {type(metrics)}")
            if not isinstance(step, (int, np.integer)):
                raise ValueError(f"Expected int for step, got {type(step)}")
            
            # Convert metrics to Python types
            converted_metrics = self._convert_batch(metrics)
            
            # Add metadata
            converted_metrics.update({
                'step': int(step),
                'timestamp': datetime.datetime.now().isoformat()
            })
            
            # Validate converted metrics
            validated_metrics = {}
            for key, value in converted_metrics.items():
                try:
                    # Skip None values
                    if value is None:
                        continue
                    
                    # Ensure metric names are strings
                    if not isinstance(key, str):
                        logger.warning(f"Non-string metric key {key} converted to string")
                        key = str(key)
                    
                    # Handle special cases
                    if isinstance(value, (list, tuple)):
                        # Ensure all elements are numeric
                        value = [float(v) if v is not None else 0.0 for v in value]
                    elif not isinstance(value, (int, float, str, bool)):
                        logger.warning(f"Unexpected type for metric {key}: {type(value)}")
                        continue
                    
                    validated_metrics[key] = value
                    
                except Exception as e:
                    logger.error(f"Error validating metric {key}: {e}")
                    continue
            
            # Log to wandb
            wandb.log(validated_metrics)
            
            # Update tables based on metric type
            try:
                if any(k.startswith('episode/') for k in validated_metrics.keys()):
                    self._update_episode_tables(validated_metrics)
                if any(k.startswith('train/') for k in validated_metrics.keys()):
                    self._update_training_tables(validated_metrics)
                if any(k.startswith('action/') for k in validated_metrics.keys()):
                    self._update_action_tables(validated_metrics)
            except Exception as e:
                logger.error(f"Error updating tables: {e}")
            
            # Update custom visualizations
            try:
                self._update_visualizations(validated_metrics)
            except Exception as e:
                logger.error(f"Error updating visualizations: {e}")
            
            if self.debug:
                logger.debug(f"Successfully logged metrics at step {step}")
            
        except Exception as e:
            logger.error(f"Critical error in _log_metrics: {e}")
            logger.error(f"Original metrics: {metrics}")
            logger.error(f"Step: {step}")
            # Don't raise to avoid breaking training