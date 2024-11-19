""" 
Based on PureJaxRL Implementation of PPO
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import jaxmarl
from jaxmarl.wrappers.action_aware import ActionAwareWrapper
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.wrappers.wandb_logger import WandbMonitorWrapper
from jaxmarl.environments.overcooked import Overcooked
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
import hydra
from omegaconf import OmegaConf
import wandb
import copy
import logging

import matplotlib.pyplot as plt

# Set up logging at the top of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CNN(nn.Module):
    activation: str = "tanh"
    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            features=64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = activation(x)

        return x


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        
        embedding = CNN(self.activation)(x)

        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def get_rollout(params, config):
    """Get rollout with consistent environment setup."""
    env = create_env_with_wrappers(config["ENV_KWARGS"], config["SEED"])
    
    network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    done = False

    obs, state = env.reset(key_r)
    # Store original state for visualization
    state_seq = [state.env_state]  # Access the underlying env_state
    
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        # Stack observations
        obs_batch = jnp.stack([obs[a] for a in env.agents])
        print("Observation batch shape:", obs_batch.shape)  # Debug print

        pi, value = network.apply(params, obs_batch)
        action = pi.sample(seed=key_a0)
        env_act = unbatchify(
            action, env.agents, 1, env.num_agents
        )

        env_act = {k: v.squeeze() for k, v in env_act.items()}

        # STEP ENV
        obs, state, reward, done, info = env.step(key_s, state, env_act)
        done = done["__all__"]
        # Store original state for visualization
        state_seq.append(state.env_state)  # Access the underlying env_state

    return state_seq


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def create_env_with_wrappers(env_kwargs, seed, use_wandb=False, wandb_config=None):
    """Create environment with consistent wrapper order."""
    # Create base environment
    env = Overcooked(**env_kwargs)
    
    # Add wrappers in consistent order
    env = LogWrapper(env, replace_info=False)  # Add logging first
    env = ActionAwareWrapper(env)  # Then add action awareness
    
    # Optionally add WandB monitoring
    if use_wandb and wandb_config:
        env = WandbMonitorWrapper(
            env,
            experiment_name=wandb_config.get("experiment_name", f"overcooked_{seed}"),
            project=wandb_config.get("project"),
            entity=wandb_config.get("entity"),
            tags=wandb_config.get("tags", []),
            group=wandb_config.get("group"),
            config=wandb_config.get("config"),
        )
    
    return env
    
def make_env(env_kwargs, seed):
    """Create environment with proper wrapper order."""
    return create_env_with_wrappers(env_kwargs, seed, use_wandb=True)


def compute_gae(traj_batch, last_val, config):
    """Compute Generalized Advantage Estimation (GAE).
    
    Args:
        traj_batch: Trajectory batch containing rewards, values, and dones
        last_val: The value estimate for the last state
        config: Config dict containing GAMMA and GAE_LAMBDA parameters
    
    Returns:
        tuple: (advantages, returns) as JAX arrays
    """
    # Get values from config
    gamma = config["GAMMA"]
    gae_lambda = config["GAE_LAMBDA"]
    
    # Ensure proper shapes for vectorized operations
    rewards = jnp.array(traj_batch.reward)  # Shape (T, B)
    values = jnp.array(traj_batch.value)    # Shape (T, B)
    dones = jnp.array(traj_batch.done)      # Shape (T, B)
    
    # Append last_val to values for computing deltas
    values_next = jnp.concatenate([values[1:], last_val[None, :]], axis=0)  # Shape (T, B)
    
    # Compute TD-error
    deltas = rewards + gamma * values_next * (1 - dones) - values
    
    def _get_advantages(carry, transition):
        """Compute advantages for a single timestep."""
        gae = carry
        delta, done = transition
        
        # Update GAE
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        
        return gae, gae

    # Compute advantages using scan (more efficient than Python loop)
    _, advantages = jax.lax.scan(
        _get_advantages,
        jnp.zeros_like(last_val),  # Initial GAE
        (deltas, dones),
        reverse=True,
    )
    
    # Compute returns from advantages
    returns = advantages + values
    
    return advantages, returns


def check_convergence(returns_history, 
                     window_size=100,        # Number of episodes to look at
                     threshold=0.01,         # Relative improvement threshold
                     patience=50):           # Number of windows to wait
    """Check if training has converged based on returns history."""
    if len(returns_history) < window_size * 2:
        return False
        
    def get_window_mean(window):
        return jnp.mean(jnp.array(window))
    
    # Compare consecutive windows
    windows_no_improvement = 0
    for i in range(len(returns_history) - window_size * 2):
        window1 = returns_history[i:i + window_size]
        window2 = returns_history[i + window_size:i + 2*window_size]
        
        mean1 = get_window_mean(window1)
        mean2 = get_window_mean(window2)
        
        # Calculate relative improvement
        relative_improvement = (mean2 - mean1) / (abs(mean1) + 1e-8)
        
        if relative_improvement < threshold:
            windows_no_improvement += 1
        else:
            windows_no_improvement = 0
            
        # Check if we've had no improvement for long enough
        if windows_no_improvement >= patience:
            return True
            
    return False


def make_train(config):
    """Create training function with environment setup and monitoring"""
    
    def train(rng):
        # Split RNG for different uses
        key, key_env, key_net = jax.random.split(rng, 3)
        
        # Create environment with consistent wrapper stack
        wandb_config = {
            "experiment_name": f"overcooked_{config['ENV_KWARGS']['layout']}_{'aa' if config['EXPERIMENT'].get('USE_ACTION_AWARE', False) else 'baseline'}",
            "project": config.get("WANDB", {}).get("PROJECT", config.get("PROJECT")),
            "entity": config.get("WANDB", {}).get("ENTITY", config.get("ENTITY")),
            "tags": config.get("WANDB", {}).get("TAGS", ["IPPO", "CNN"]) + (["action_aware"] if config['EXPERIMENT'].get('USE_ACTION_AWARE', False) else []),
            "group": config.get("WANDB", {}).get("GROUP", "overcooked_experiments"),
            "config": config,
        }
        
        env = create_env_with_wrappers(
            config["ENV_KWARGS"], 
            config["SEED"],
            use_wandb=True,
            wandb_config=wandb_config
        )
        
        # Update NUM_ACTORS based on environment
        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
        
        # Initialize network and optimizer
        network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
        
        # Get initial network parameters
        init_obs = jnp.zeros((1,) + env.observation_space().shape)
        print("Network initialization shape:", init_obs.shape)
        network_params = network.init(key_net, init_obs)

        # Create train state with optimizer
        if config["ANNEAL_LR"]:
            schedule = optax.linear_schedule(
                init_value=config["LR"],
                end_value=0.0,
                transition_steps=config["NUM_UPDATES"],
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # Initialize environment states
        obs, env_state = env.reset(key_env)
        
        # Initialize runner state
        runner_state = (
            train_state,
            env_state,
            obs,
            jnp.zeros(config["NUM_ENVS"]),  # Initial done flags
            key,
        )

        returns_history = []
        max_updates = config.get("MAX_UPDATES", 5000)
        min_updates = config.get("MIN_UPDATES", 1000)
        
        # Training loop
        def _update_step(runner_state, step_idx):
            # Unpack runner state
            train_state, env_state, obs, key = runner_state
            
            # Collect trajectories
            def _env_step(runner_state, unused):
                # Unpack runner state
                key, train_state, last_obs, last_done, env_state = runner_state

                # Split key for various random operations
                key, key_step = jax.random.split(key)

                # Get action from policy
                obs_batch = batchify(last_obs, env.agents, config["NUM_ENVS"])
                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=key_step)
                log_prob = pi.log_prob(action)
                
                # Convert actions to environment format
                env_actions = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                env_actions = {k: v.squeeze() for k, v in env_actions.items()}
                
                # Step environment
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                obs, env_state, reward, done, info = env.step(key_step, env_state, env_act)

                # Process rewards to ensure proper shape
                reward_array = batchify(reward, env.agents, config["NUM_ENVS"])

                transition = Transition(
                    done=done["__all__"],
                    action=action,
                    value=value,
                    reward=reward_array,  # Use processed reward array
                    log_prob=log_prob,
                    obs=obs_batch,
                    info=info,
                )

                # Update runner state
                runner_state = (key, train_state, obs, done["__all__"], env_state)
                return runner_state, transition
            
            # Collect trajectories for NUM_STEPS
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            
            # Update policy using PPO
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    
                    def _loss_fn(params, traj_batch, advantages, targets):
                        # Standard PPO loss computation
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)
                        
                        # Policy loss
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        clip_ratio = jnp.clip(ratio, 1 - config["CLIP_EPS"], 1 + config["CLIP_EPS"])
                        policy_loss = -jnp.mean(
                            jnp.minimum(
                                ratio * advantages,
                                clip_ratio * advantages,
                            )
                        )
                        
                        # Value loss
                        value_loss = jnp.mean(jnp.square(value - targets))
                        
                        # Entropy loss
                        entropy_loss = -jnp.mean(pi.entropy())
                        
                        # Total loss
                        total_loss = (
                            policy_loss 
                            + config["VF_COEF"] * value_loss 
                            + config["ENT_COEF"] * entropy_loss
                        )
                        
                        return total_loss, (policy_loss, value_loss, entropy_loss)
                    
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, (policy_loss, value_loss, entropy_loss)), grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    
                    train_state = train_state.apply_gradients(grads=grads)
                    
                    metrics = {
                        "total_loss": total_loss,
                        "policy_loss": policy_loss,
                        "value_loss": value_loss,
                        "entropy_loss": entropy_loss,
                    }
                    
                    return train_state, metrics
                
                train_state, traj_batch, advantages, targets = update_state
                
                # Update minibatches
                train_state, metrics = jax.lax.scan(
                    _update_minibatch,
                    train_state,
                    (traj_batch, advantages, targets),
                    config["NUM_MINIBATCHES"],
                )
                
                update_state = (train_state, traj_batch, advantages, targets)
                return update_state, metrics
            
            # Compute advantages and returns
            advantages, targets = compute_gae(
                traj_batch,
                config["GAMMA"],
                config["GAE_LAMBDA"],
            )
            
            # Update for multiple epochs
            update_state = (train_state, traj_batch, advantages, targets)
            update_state, metrics = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            
            train_state = update_state[0]
            runner_state = (train_state, env_state, obs, done, key)  # Include done flags
            
            # Track episode returns
            episode_return = metrics.get("episode_return", 0.0)
            returns_history.append(episode_return)
            
            # Check for convergence after minimum updates
            if step_idx > min_updates:
                has_converged = check_convergence(
                    returns_history,
                    window_size=config.get("CONV_WINDOW_SIZE", 100),
                    threshold=config.get("CONV_THRESHOLD", 0.01),
                    patience=config.get("CONV_PATIENCE", 50)
                )
                
                if has_converged:
                    # Signal early stopping
                    metrics["converged"] = True
                    
            return runner_state, metrics
        
        # Modified training loop with early stopping
        final_state = runner_state
        final_metrics = None
        
        for update_idx in range(max_updates):
            runner_state, metrics = _update_step(runner_state, update_idx)
            
            if metrics.get("converged", False):
                print(f"Training converged after {update_idx} updates")
                break
                
            final_state = runner_state
            final_metrics = metrics
            
        return {
            "runner_state": final_state, 
            "metrics": final_metrics,
            "updates_used": update_idx + 1,
            "returns_history": returns_history
        }
    
    return train

def run_training(config):
    """Run single training configuration and return results"""
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)
    
    # Extract results for comparison
    results = {
        "returns": jax.tree_util.tree_map(lambda x: x[0], out["metrics"]["returned_episode_returns"]),
        "dishes": jax.tree_util.tree_map(lambda x: x[0], out["metrics"].get("completed_dishes", [])),
        "collisions": jax.tree_util.tree_map(lambda x: x[0], out["metrics"].get("collisions", [])),
    }
    
    return results

def single_run(config):
    """Run both baseline and action-aware experiments"""
    try:
        config = OmegaConf.to_container(config)
        layout_name = copy.deepcopy(config["ENV_KWARGS"]["layout"])
        config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]
        
        results = {}
        
        # Run baseline if configured
        if config["EXPERIMENT"]["RUN_BASELINE"]:
            logger.info("Starting baseline run...")
            baseline_config = copy.deepcopy(config)
            baseline_config["EXPERIMENT"]["USE_ACTION_AWARE"] = False
            results["baseline"] = run_training(baseline_config)
            logger.info("Completed baseline run")
        
        # Run action-aware
        logger.info("Starting action-aware run...")
        action_config = copy.deepcopy(config)
        action_config["EXPERIMENT"]["USE_ACTION_AWARE"] = True
        results["action_aware"] = run_training(action_config)
        logger.info("Completed action-aware run")
        
        # Log comparison metrics
        try:
            steps = range(len(results["baseline"]["returns"]))
            wandb.log({
                "comparison/returns": wandb.plot.line_series(
                    xs=steps,
                    ys=[results["baseline"]["returns"], results["action_aware"]["returns"]],
                    keys=["Baseline", "Action-Aware"],
                    title="Return Comparison",
                    xname="steps"
                ),
                "comparison/dishes": wandb.plot.line_series(
                    xs=steps,
                    ys=[results["baseline"]["dishes"], results["action_aware"]["dishes"]],
                    keys=["Baseline", "Action-Aware"],
                    title="Completed Dishes Comparison",
                    xname="steps"
                ),
                "comparison/collisions": wandb.plot.line_series(
                    xs=steps,
                    ys=[results["baseline"]["collisions"], results["action_aware"]["collisions"]],
                    keys=["Baseline", "Action-Aware"],
                    title="Agent Collisions Comparison",
                    xname="steps"
                )
            })
            logger.info("Successfully logged comparison metrics to wandb")
        except Exception as e:
            logger.error(f"Failed to log comparison metrics: {e}")
        
        # Generate visualizations
        try:
            for run_type, run_results in results.items():
                filename = f'{config["ENV_NAME"]}_{layout_name}_{run_type}_seed{config["SEED"]}'
                train_state = jax.tree_util.tree_map(lambda x: x[0], run_results["runner_state"][0])
                state_seq = get_rollout(train_state.params, config)
                viz = OvercookedVisualizer()
                viz.animate(state_seq, agent_view_size=5, filename=f"{filename}.gif")
            logger.info("Successfully generated visualizations")
        except Exception as e:
            logger.error(f"Failed to generate visualization: {e}")
            
        return results
        
    except Exception as e:
        logger.error(f"Critical error in single_run: {e}")
        raise


def tune(default_config):
    """Hyperparameter sweep with wandb."""
    import copy

    default_config = OmegaConf.to_container(default_config)

    layout_name = default_config["ENV_KWARGS"]["layout"]

    def wrapped_make_train():

        wandb.init(project=default_config["PROJECT"])
        # update the default params
        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v
        config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]

        print("running experiment with params:", config)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))

    sweep_config = {
        "name": "ppo_overcooked",
        "method": "bayes",
        "metric": {
            "name": "returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "NUM_ENVS": {"values": [32, 64, 128, 256]},
            "LR": {"values": [0.0005, 0.0001, 0.00005, 0.00001]},
            "ACTIVATION": {"values": ["relu", "tanh"]},
            "UPDATE_EPOCHS": {"values": [2, 4, 8]},
            "NUM_MINIBATCHES": {"values": [2, 4, 8, 16]},
            "CLIP_EPS": {"values": [0.1, 0.2, 0.3]},
            "ENT_COEF": {"values": [0.0001, 0.001, 0.01]},
            "NUM_STEPS": {"values": [64, 128, 256]},
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


@hydra.main(version_base=None, config_path="config", config_name="ippo_cnn_overcooked_aa_wandb")
def main(config):
    if config["TUNE"]:
        tune(config)
    else:
        single_run(config)

if __name__ == "__main__":
    main()
