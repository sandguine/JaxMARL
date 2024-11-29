"""
Implementation of Independent PPO (IPPO) for multi-agent environments.
Based on PureJaxRL's PPO implementation but adapted for multi-agent scenarios.
"""

# Core imports for JAX machine learning
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax

# Environment and visualization imports
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer

# Configuration and logging imports
import hydra
from omegaconf import OmegaConf
import wandb

import matplotlib.pyplot as plt


class ActorCritic(nn.Module):
    """Neural network architecture implementing both policy (actor) and value function (critic)"""
    action_dim: Sequence[int]  # Dimension of action space
    activation: str = "tanh"   # Activation function to use

    @nn.compact
    def __call__(self, x):
        print("network input x shape:", x.shape)  # Debug print for input shape
        
        # Select activation function based on config
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Actor network - outputs action probabilities
        print("ActorCritic input shape:", x.shape)

        # Check input dimensions
        if len(x.shape) == 1:
            expected_dim = x.shape[0]
        else:
            expected_dim = x.shape[-1]
        print(f"Expected input dim: {expected_dim}")  # Debug print


        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        # Final layer outputs logits for each possible action
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        # Convert logits to categorical distribution
        pi = distrax.Categorical(logits=actor_mean)

        # Critic network - outputs value function estimate
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        # Final layer outputs single value estimate
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)
    
class Transition(NamedTuple):
    """Container for storing experience transitions"""
    done: jnp.ndarray      # Episode termination flag
    action: jnp.ndarray    # Action taken
    value: jnp.ndarray     # Value function estimate
    reward: jnp.ndarray    # Reward received
    log_prob: jnp.ndarray  # Log probability of action
    obs: jnp.ndarray       # Observation

def get_rollout(train_state, config):
    """Generate a single episode rollout for visualization.
    
    Runs a single episode in the environment using the current policy networks to generate
    actions. Used for visualizing agent behavior during training.
    
    Args:
        train_state: Current training state containing network parameters
        config: Dictionary containing environment and training configuration
        
    Returns:
        Episode trajectory data including states, rewards, and shaped rewards
    """
    # Initialize environment
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    # env_params = env.default_params
    # env = LogWrapper(env)

    # Initialize network
    network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
    
    # Initialize seeds
    key = jax.random.PRNGKey(0)
    key, key_a, key_r = jax.random.split(key, 3)
    # Split key_a for agent_1/agent_0 network init
    key_a_agent_0, key_a_agent_1 = jax.random.split(key_a)
    # key_r for environment reset
    # key for future episode steps

    # Get observation shapes
    base_obs_shape = env.observation_space().shape
    print("base_obs_shape:", base_obs_shape)
    print("action_space:", env.action_space().n)
    ego_obs_shape = (*base_obs_shape[:-1], base_obs_shape[-1] + env.action_space().n)
    print("ego_obs_shape:", ego_obs_shape)
    
    # Initialize networks with appropriate shapes
    init_x_agent_0 = jnp.zeros(ego_obs_shape).flatten()
    init_x_agent_1 = jnp.zeros(base_obs_shape).flatten()
    
    network_params_agent_0 = network.init(key_a_agent_0, init_x_agent_0)
    network_params_agent_1 = network.init(key_a_agent_1, init_x_agent_1)

    done = False

    # Reset environment using key_r
    obs, state = env.reset(key_r)
    state_seq = [state]
    rewards = []
    shaped_rewards = []

    # Run episode until completion
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        # obs_batch = batchify(obs, env.agents, config["NUM_ACTORS"])
        # breakpoint()

        # Flatten observations for network input
        obs = {k: v.flatten() for k, v in obs.items()}

        print("agent_0 obs shape:", obs["agent_0"].shape)
        print("agent_1 obs shape:", obs["agent_1"].shape)

        # Get actions from policy for both agents
        pi_0, _ = network.apply(network_params_agent_0, obs["agent_0"])
        pi_1, _ = network.apply(network_params_agent_1, obs["agent_1"])

        actions = {"agent_0": pi_0.sample(seed=key_a0), "agent_1": pi_1.sample(seed=key_a1)}
        print("actions:", actions.shape)
        # env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
        # env_act = {k: v.flatten() for k, v in env_act.items()}

        # Step environment forward
        obs, state, reward, done, info = env.step(key_s, state, actions)
        print("shaped reward:", info["shaped_reward"])
        done = done["__all__"]
        rewards.append(reward['agent_0'])
        shaped_rewards.append(info["shaped_reward"]['agent_0'])

        state_seq.append(state)

    # Plot rewards for visualization
    from matplotlib import pyplot as plt

    plt.plot(rewards, label="reward")
    plt.plot(shaped_rewards, label="shaped_reward")
    plt.legend()
    plt.savefig("reward.png")
    plt.show()

    return state_seq

def batchify(x: dict, agent_list, num_actors):
    """Batchify observations for a single agent.
    
    This function stacks the observations for a single agent across multiple environments,
    reshaping them into a single array with shape (num_actors, -1).

    Args:
        x: Dictionary containing observations for a single agent
        agent_list: List of agent names
        num_actors: Number of parallel environments

    Returns:
        Batched observations with shape (num_actors, -1)
    """
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    """Convert batched array back to dict of agent observations.
    
    This function reshapes the batched array back to a dictionary of observations for
    each agent, with shape (num_envs, -1).

    Args:
        x: Batched observations with shape (num_actors, num_envs, -1)
        agent_list: List of agent names
        num_envs: Number of parallel environments
        num_actors: Number of actors (agents)

    Returns:
        Dictionary containing observations for each agent
    """
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(config):
    """Creates the main training function for IPPO with the given configuration.
    
    This function sets up the training environment, networks, and optimization process
    for training multiple agents using Independent PPO (IPPO). It handles:
    - Environment initialization and wrapping
    - Network architecture setup for both agents
    - Learning rate scheduling and reward shaping annealing
    - Training loop configuration including batch sizes and update schedules
    
    Args:
        config: Dictionary containing training hyperparameters and environment settings
               including:
               - ENV_NAME: Name of environment to train in
               - ENV_KWARGS: Environment configuration parameters
               - NUM_ENVS: Number of parallel environments
               - NUM_STEPS: Number of steps per training iteration
               - TOTAL_TIMESTEPS: Total environment steps to train for
               - Learning rates, batch sizes, and other optimization parameters
               
    Returns:
        train: The main training function that takes an RNG key and executes the full
               training loop, returning the trained agent policies
    """
    
    # Initialize environment
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # Calculate key training parameters
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"] 
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # Configuration printing
    print("Initializing training with config:")
    print(f"NUM_ENVS: {config['NUM_ENVS']}")
    print(f"NUM_STEPS: {config['NUM_STEPS']}")
    print(f"NUM_UPDATES: {config['NUM_UPDATES']}")
    print(f"NUM_MINIBATCHES: {config['NUM_MINIBATCHES']}")
    print(f"TOTAL_TIMESTEPS: {config['TOTAL_TIMESTEPS']}")
    
    env = LogWrapper(env, replace_info=False)
    
    def linear_schedule(count):
        """Linear learning rate annealing schedule that decays over training.
        
        Calculates a learning rate multiplier that decreases linearly from 1.0 to 0.0
        over the course of training. Used to gradually reduce the learning rate to help
        convergence.
        
        Args:
            count: Current training step count used to calculate progress through training
        
        Returns:
            float: The current learning rate after applying the annealing schedule,
                  calculated as: base_lr * (1 - training_progress)
        """
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac
    
    # Schedule for annealing reward shaping
    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.,
        end_value=0.,
        transition_steps=config["REW_SHAPING_HORIZON"]
    )

    def train(rng):
        """Main training loop for Independent PPO (IPPO) algorithm.
        
        Implements the core training loop for training multiple agents using IPPO.
        Handles network initialization, environment setup, and training iteration.
        
        Args:
            rng: JAX random number generator key for reproducibility
            
        Returns:
            Tuple containing:
            - Final trained network parameters for both agents
            - Training metrics and statistics
            - Environment states from training
            
        The training process:
        1. Initializes separate policy networks for each agent
        2. Collects experience in parallel environments
        3. Updates policies using PPO with independent value functions
        4. Tracks and logs training metrics
        """
        # Shapes we're initializing with
        print("Action space:", env.action_space().n)
        print("Observation space shape:", env.observation_space().shape)

        # Initialize network and optimizer
        network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])

        # Initialize seeds
        rng, _rng = jax.random.split(rng)
        _rng_agent_0, _rng_agent_1 = jax.random.split(_rng)  # Split for two networks
        
        # Get observation shapes
        base_obs_shape = env.observation_space().shape
        base_obs_dim = np.prod(base_obs_shape)
        ego_obs_shape = base_obs_dim + env.action_space().n
        
        # Initialize networks with appropriate shapes
        init_x_agent_0 = jnp.zeros(ego_obs_shape).flatten()
        init_x_agent_1 = jnp.zeros(base_obs_shape).flatten()
        
        network_params_agent_0 = network.init(_rng_agent_0, init_x_agent_0)
        network_params_agent_1 = network.init(_rng_agent_1, init_x_agent_1)
        
        def create_optimizer(config):
            """Creates an optimizer chain for training each agent's neural network.
            
            The optimizer chain consists of:
            1. Gradient clipping using global norm
            2. Adam optimizer with either:
            - Annealed learning rate that decays linearly over training
            - Fixed learning rate specified in config
            
            Args:
                config: Dictionary containing optimization parameters like:
                    - ANNEAL_LR: Whether to use learning rate annealing
                    - MAX_GRAD_NORM: Maximum gradient norm for clipping
                    - LR: Base learning rate
                    
            Returns:
                optax.GradientTransformation: The composed optimizer chain
            """
            if config["ANNEAL_LR"]:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),  # First transformation
                    optax.adam(learning_rate=linear_schedule, eps=1e-5)  # Second transformation
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(config["LR"], eps=1e-5)
                )
            return tx

        # Create separate optimizer chains for each agent
        tx_agent_0 = create_optimizer(config)
        tx_agent_1 = create_optimizer(config)

        # Create separate train states
        train_state = {
            'agent_0': TrainState.create(
                apply_fn=network.apply,
                params=network_params_agent_0,
                tx=tx_agent_0
            ),
            'agent_1': TrainState.create(
                apply_fn=network.apply,
                params=network_params_agent_1,
                tx=tx_agent_1
            )
        }
        
        # Initialize environment states
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        
        # TRAIN LOOP
        def _update_step(runner_state, unused):
            """Executes a single training update step in the IPPO algorithm.
            
            This function performs one complete update iteration including:
            1. Collecting trajectories by running the current policy in the environment
            2. Computing advantages and returns
            3. Updating both agents' neural networks using PPO
            
            The update handles both agents (agent_0 and agent_1) separately, with agent_0
            receiving augmented observations that include agent_1's actions.
            
            Args:
                runner_state: Tuple containing:
                    - train_state: Current training state with network parameters
                    - env_state: Current environment state
                    - last_obs: Previous observations from environment
                    - update_step: Current training iteration number
                    - rng: Random number generator state
                unused: Placeholder parameter for JAX scan compatibility
                
            Returns:
                Tuple containing:
                    - Updated runner_state
                    - Metrics dictionary with training statistics
            """
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                """Collects trajectories by running the current policy in the environment.
                
                This function performs one step of environment interaction for each agent,
                collecting trajectories for training.

                Args:
                    runner_state: Tuple containing:
                        - train_state: Current training state with network parameters
                        - env_state: Current environment state
                        - last_obs: Previous observations from environment
                        - update_step: Current training iteration number
                        - rng: Random number generator state

                    unused: Placeholder parameter for JAX scan compatibility
                    
                Returns:
                    Tuple containing:
                        - Updated runner_state
                        - Trajectory batch, info, and processed observations
                """
                train_state, env_state, last_obs, update_step, rng = runner_state

                # SELECT ACTION consistently with training initialization
                rng, _rng = jax.random.split(rng)
                _rng_agent_0, _rng_agent_1 = jax.random.split(_rng)  # Split for two networks

                # agent_1 step - using original observation
                print("\nAgent_1 shapes:")
                print("Original agent_1 obs shape:", last_obs['agent_1'].shape)
                agent_1_obs = last_obs['agent_1'].reshape(last_obs['agent_1'].shape[0], -1)
                print("agent_1 obs shape:", agent_1_obs.shape)
                agent_1_pi, agent_1_value = network.apply(train_state.params['agent_1'], agent_1_obs)
                print("agent_1_value shape:", agent_1_value.shape)
                agent_1_action = agent_1_pi.sample(seed=_rng_agent_1)
                print("agent_1 action shape:", agent_1_action.shape)
                agent_1_log_prob = agent_1_pi.log_prob(agent_1_action)
                print("agent_1 log prob shape:", agent_1_log_prob.shape)

                # agent_0 step - augmenting observation with agent_1 action
                print("\nAgent_0 shapes:")
                print("Original agent_0 obs shape:", last_obs['agent_0'].shape)
                one_hot_action = jax.nn.one_hot(agent_1_action, env.action_space().n)
                # Flatten observation while preserving batch dimension
                agent_0_obs = last_obs['agent_0'].reshape(last_obs['agent_0'].shape[0], -1)
                print("agent_0 obs shape:", agent_0_obs.shape)
                agent_0_obs_augmented = jnp.concatenate([
                    agent_0_obs,
                    one_hot_action
                ], axis=-1)
                print("agent_0_obs_augmented shape:", agent_0_obs_augmented.shape)
                agent_0_pi, agent_0_value = network.apply(train_state.params['agent_0'], agent_0_obs_augmented)
                print("agent_0_value shape:", agent_0_value.shape)
                agent_0_action = agent_0_pi.sample(seed=_rng_agent_0)
                print("agent_0_action_shape:", agent_0_action.shape)
                agent_0_log_prob = agent_0_pi.log_prob(agent_0_action)
                print("agent_0_log_prob_shape:", agent_0_log_prob.shape)

                processed_obs = {
                    'agent_0': agent_0_obs_augmented,
                    'agent_1': agent_1_obs
                }
                print("\nProcessed obs shapes:")
                print("Processed agent_0 obs shape:", processed_obs['agent_0'].shape)
                print("Processed agent_1 obs shape:", processed_obs['agent_1'].shape)

                # Package actions for environment step
                env_act = {
                    "agent_0": agent_0_action,
                    "agent_1": agent_1_action
                }
                print("\nEnv act shapes before flattening:")
                print("Env act agent_0 shape:", env_act['agent_0'].shape)
                print("Env act agent_1 shape:", env_act['agent_1'].shape)
                env_act = {k:v.flatten() for k,v in env_act.items()}
                print("\nEnv act shapes after flattening:")
                print("Env act agent_0 shape:", env_act['agent_0'].shape)
                print("Env act agent_1 shape:", env_act['agent_1'].shape)
                
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    rng_step, env_state, env_act
                )
                print("next_obs structure:", jax.tree_map(lambda x: x.shape, obsv))

                
                # Store original reward for logging
                info["reward"] = reward["agent_0"]

                # Apply reward shaping
                current_timestep = update_step*config["NUM_STEPS"]*config["NUM_ENVS"]
                reward = jax.tree.map(lambda x,y: x+y*rew_shaping_anneal(current_timestep), reward, info["shaped_reward"])
                print("shaped reward:", info["shaped_reward"])

                transition = Transition(
                    done=jnp.array([done["agent_1"], done["agent_0"]]).squeeze(),
                    action=jnp.array([agent_1_action, agent_0_action]),
                    value=jnp.array([agent_1_value, agent_0_value]),
                    reward=jnp.array([
                        reward["agent_1"],  # Agent 1's rewards
                        reward["agent_0"]   # Agent 0's rewards
                    ]).squeeze(),
                    log_prob=jnp.array([agent_1_log_prob, agent_0_log_prob]),
                    obs=processed_obs
                )

                runner_state = (train_state, env_state, obsv, update_step, rng)
                return runner_state, (transition, info, processed_obs)

            runner_state, (traj_batch, info, processed_obs) = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            
            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, update_step, rng = runner_state
            
            # Calculate last values for both agents
            print("\nCalculating last values:")
            last_obs_agent1 = last_obs['agent_1'].reshape(last_obs['agent_1'].shape[0], -1)
            _, agent_1_last_val = network.apply(train_state.params['agent_1'], last_obs_agent1)
            print("agent_1_last_val shape:", agent_1_last_val.shape)

            # For agent_0, need to include agent_1's last action in observation
            one_hot_last_action = jax.nn.one_hot(traj_batch.action[-1, 1], env.action_space().n)
            last_obs_agent0 = last_obs['agent_0'].reshape(last_obs['agent_0'].shape[0], -1)
            last_obs_agent0_augmented = jnp.concatenate([last_obs_agent0, one_hot_last_action], axis=-1)
            _, agent_0_last_val = network.apply(train_state.params['agent_0'], last_obs_agent0_augmented)
            print("agent_0_last_val shape:", agent_0_last_val.shape)

            # Combine values for advantage calculation
            last_val = jnp.array([agent_1_last_val, agent_0_last_val])
            print("stacked last_val shape:", last_val.shape)

            # calculate_gae itself didn't need to be changed because we can use the same advantage function for both agents
            def _calculate_gae(traj_batch, last_val):
                """Calculate Generalized Advantage Estimation (GAE) for trajectories.
                
                This function computes the GAE for a given trajectory batch and last value,
                which are used to estimate the advantage of each action in the trajectory.

                Args:
                    traj_batch: Trajectory batch containing transitions
                    last_val: Last value estimates for the trajectory
                    
                Returns:
                    Tuple containing:
                        - Advantages for the trajectory
                        - Returns (advantages + value estimates)
                """
                print(f"\nGAE Calculation Debug:")
                print("traj_batch types:", jax.tree_map(lambda x: x.dtype, traj_batch))
                print(f"traj_batch shapes:", jax.tree_map(lambda x: x.shape, traj_batch))
                print("last_val types:", jax.tree_map(lambda x: x.dtype, last_val))
                print(f"last_val shape: {last_val.shape}")
                

                def _get_advantages(gae_and_next_value, transition):
                    """Calculate GAE and returns for a single transition.
                    
                    This function computes the GAE and returns for a single transition,
                    which are used to update the policy and value functions.
                    
                    Args:
                        gae_and_next_value: Tuple containing current GAE and next value
                        transition: Single transition containing data for one step
                    
                    Returns:
                        Tuple containing:
                            - Updated GAE and next value
                            - Calculated GAE
                    """
                    gae, next_value = gae_and_next_value
                    
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )

                    # Debug intermediate calculations
                    print(f"\nGAE step debug:")
                    print(f"done shape: {done.shape}")
                    print(f"value shape: {value.shape}")
                    print(f"reward shape: {reward.shape}")
                    print(f"next_value shape: {next_value.shape}")
                    print(f"gae shape: {gae.shape}")

        
                     # Calculate delta and GAE per agent
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    print(f"delta shape: {delta.shape}, value: {delta}")

                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    print(f"calculated gae shape: {gae.shape}, value: {gae}")
                    
                    return (gae, value), gae

                # Initialize with agent-specific last value
                init_gae = jnp.zeros_like(last_val)
                init_value = last_val

                # Calculate advantages for an agent
                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (init_gae, init_value),
                    traj_batch,
                    reverse=True,
                    unroll=16
                )

                # Calculate returns (advantages + value estimates)
                print(f"\nFinal shapes:")
                print(f"advantages shape: {advantages.shape}")
                print(f"returns shape: {(advantages + traj_batch.value).shape}")
                return advantages, advantages + traj_batch.value

            # Calculate advantages 
            advantages, targets = _calculate_gae(traj_batch, last_val)
            print("advantages shape:", advantages.shape)
            print("targets shape:", targets.shape)
            print("traj_batch value shape:", traj_batch.value.shape)
            print("traj_batch reward shape:", traj_batch.reward.shape)
            print("traj_batch data types:", jax.tree_map(lambda x: x.dtype, traj_batch))

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                """
                Performs a complete training epoch for both agents.
                
                Args:
                    update_state: Tuple containing (train_state, traj_batch, advantages, targets, rng)
                    unused: Placeholder for scan compatibility
                    
                Returns:
                    Updated state and loss information
                """
                def _update_minbatch(train_state, batch_info):
                    """Updates network parameters using a minibatch of experience.
                    
                    Args:
                        train_state: Current training state containing both agents' parameters
                        batch_info: Tuple of (traj_batch, advantages, targets) for both agents
                        
                    Returns:
                        Updated training state and loss information
                    """
                    print("\nStarting minibatch update...")
                    # Unpack batch_info which now contains separate agent data
                    agent_0_data, agent_1_data = batch_info['agent_0'], batch_info['agent_1']
                    
                    print("Minibatch shapes:")
                    print("Agent 0 data:", jax.tree_map(lambda x: x.shape, agent_0_data))
                    print("Agent 1 data:", jax.tree_map(lambda x: x.shape, agent_1_data))


                    # Loss function itself didn't need to be changed because we can use the same loss function for both agents
                    def _loss_fn(params, traj_batch, gae, targets):
                        """Calculate loss for a single agent.
                        
                        This function computes the loss for a single agent, which is used
                        to update the policy and value functions.
                        
                        Args:
                            params: Network parameters for the agent
                            traj_batch: Trajectory batch containing transitions
                            gae: Generalized Advantage Estimation (GAE) for the trajectory
                            targets: Target values (advantages + value estimates) for the trajectory
                        
                        Returns:
                            Tuple containing:
                                - Total loss for the agent
                                - Auxiliary loss information (value loss, actor loss, entropy)
                        """
                        # RERUN NETWORK
                        print("\nCalculating losses...")
                        pi, value = network.apply(params, traj_batch.obs)
                        print(f"Network outputs - pi shape: {pi.batch_shape}, value shape: {value.shape}")
                        log_prob = pi.log_prob(traj_batch.action)
                        print(f"Log prob shape: {log_prob.shape}")

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        print(f"Value loss: {value_loss}")

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        print(f"Importance ratio shape: {ratio.shape}")
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        print(f"Normalized GAE shape: {gae.shape}")
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()
                        print(f"Actor loss: {loss_actor}, Entropy: {entropy}")

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        print(f"Total loss: {total_loss}")
                        return total_loss, (value_loss, loss_actor, entropy)

                    # Create gradient function for both agents
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
    
                    # Compute gradients for agent 0
                    (loss_0, aux_0), grads_0 = grad_fn(
                        train_state.params['agent_0'],
                        agent_0_data['traj'],
                        agent_0_data['advantages'],
                        agent_0_data['targets']
                    )
    
                    # Compute gradients for agent 1
                    (loss_1, aux_1), grads_1 = grad_fn(
                        train_state.params['agent_1'],
                        agent_1_data['traj'],
                        agent_1_data['advantages'],
                        agent_1_data['targets']
                    )
    
                    print("\nGradient stats:")
                    print(f"Grad norm agent_0: {optax.global_norm(grads_0)}")
                    print(f"Grad norm agent_1: {optax.global_norm(grads_1)}")
                    
                    # Update both agents' parameters separately
                    train_state = train_state.replace(
                        params={
                            'agent_0': train_state.tx.update(grads_0, train_state.params['agent_0'])[0],
                            'agent_1': train_state.tx.update(grads_1, train_state.params['agent_1'])[0]
                        }
                    )
    
                    # Combine losses for logging
                    total_loss = loss_0 + loss_1
                    combined_aux = jax.tree_map(lambda x, y: (x + y) / 2, aux_0, aux_1)
                    
                    return train_state, (total_loss, combined_aux)

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                # Calculate total batch size and minibatch size
                batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                config["MINIBATCH_SIZE"] = batch_size // config["NUM_MINIBATCHES"]

                # Verify that the data can be evenly split into minibatches
                assert (
                    batch_size % config["NUM_MINIBATCHES"] == 0
                ), "Steps * Envs must be divisible by number of minibatches"

                # Separate data for each agent, separate the individual components of the transitions explicitly 
                # to handle different observation size
                agent_data = {
                    'agent_0': {
                    'traj': Transition(
                        done=traj_batch.done[:, 0],           # Shape: (128, 16)
                        action=traj_batch.action[:, 0],       # Shape: (128, 16)
                        value=traj_batch.value[:, 0],         # Shape: (128, 16)
                        reward=traj_batch.reward[:, 0],       # Shape: (128, 16)
                        log_prob=traj_batch.log_prob[:, 0],   # Shape: (128, 16)
                        obs=traj_batch.obs['agent_0']         # Shape: (128, 526)
                    ),
                    'advantages': advantages[:, 0],           # Shape: (128, 16)
                    'targets': targets[:, 0]                  # Shape: (128, 16)
                },
                    'agent_1': {
                        'traj': Transition(
                            done=traj_batch.done[:, 1],           # Shape: (128, 16)
                            action=traj_batch.action[:, 1],       # Shape: (128, 16)
                            value=traj_batch.value[:, 1],         # Shape: (128, 16)
                            reward=traj_batch.reward[:, 1],       # Shape: (128, 16)
                            log_prob=traj_batch.log_prob[:, 1],   # Shape: (128, 16)
                            obs=traj_batch.obs['agent_1']         # Shape: (128, 520)
                        ),
                        'advantages': advantages[:, 1],           # Shape: (128, 16)
                        'targets': targets[:, 1]                  # Shape: (128, 16)
                    }
                }

                print("\nBatch processing, Pre-reshape diagnostics:")
                print("agent_0 obs structure:", jax.tree_map(lambda x: x.shape if hasattr(x, 'shape') else type(x), agent_data['agent_0']['traj'].obs))
                print("agent_1 obs structure:", jax.tree_map(lambda x: x.shape if hasattr(x, 'shape') else type(x), agent_data['agent_1']['traj'].obs))
                print("Advantages shape:", advantages.shape)
                print("Targets shape:", targets.shape)

                # Reshape function that handles the different observation sizes
                def reshape_agent_data(agent_dict):
                    """Reshape trajectory data for a single agent.
                    
                    This function reshapes the trajectory data for a single agent,
                    ensuring that the observations are reshaped correctly while keeping
                    the features dimension intact.
                    
                    Args:
                        agent_dict: Dictionary containing trajectory data for an agent
                    
                    Returns:
                        Dictionary containing reshaped trajectory data
                    """
                    def reshape_field(x, field_name):
                        """Reshape a single field of the trajectory data.
                        
                        This function reshapes a single field of the trajectory data,
                        ensuring that the observations are reshaped correctly while keeping
                        the features dimension intact.

                        Args:
                            x: Data to be reshaped
                            field_name: Name of the field to be reshaped
                        
                        Returns:
                            Reshaped data
                        """
                        if not isinstance(x, (dict, jnp.ndarray)):
                            return x
                            
                        if field_name == 'obs':
                            # Keep the features dimension intact, only combine timesteps and envs
                            print(f"Reshaping {field_name} from {x.shape} to {(batch_size, -1)}")
                            return x.reshape(batch_size, -1)
                        else:
                            # For other fields, flatten to (batch_size,)
                            print(f"Reshaping {field_name} from {x.shape} to {(batch_size,)}")
                            return x.reshape(batch_size)
                            
                    return {
                        'traj': Transition(
                            **{field: reshape_field(getattr(agent_dict['traj'], field), field)
                               for field in agent_dict['traj']._fields}
                        ),
                        'advantages': agent_dict['advantages'].reshape(batch_size),
                        'targets': agent_dict['targets'].reshape(batch_size)
                    }

                # Reshape each agent's data
                agent_data = {
                    agent: reshape_agent_data(data)
                    for agent, data in agent_data.items()
                }

                # After reshaping:
                print("\nPost-reshape diagnostics:")
                print("Reshaped agent_0 obs:", jax.tree_map(lambda x: x.shape if hasattr(x, 'shape') else type(x), agent_data['agent_0']['traj'].obs))
                print("Reshaped agent_1 obs:", jax.tree_map(lambda x: x.shape if hasattr(x, 'shape') else type(x), agent_data['agent_1']['traj'].obs))

                # Create permutation and shuffle
                permutation = jax.random.permutation(_rng, batch_size)
                agent_data = {
                    agent: {
                        'traj': Transition(
                            **{field: jnp.take(getattr(data['traj'], field), permutation, axis=0)
                               for field in data['traj']._fields}
                        ),
                        'advantages': jnp.take(data['advantages'], permutation, axis=0),
                        'targets': jnp.take(data['targets'], permutation, axis=0)
                    }
                    for agent, data in agent_data.items()
                }

                print("\nShuffled batch structure:", jax.tree_map(lambda x: x.shape if hasattr(x, 'shape') else type(x), agent_data))

                # Create minibatches
                def create_minibatches(data):
                    """Create minibatches from trajectory data.
                    
                    This function divides the trajectory data into smaller minibatches,
                    which are used for training the policy and value networks.
                    
                    Args:
                        data: Dictionary containing trajectory data for an agent
                    
                    Returns:
                        Dictionary containing minibatched trajectory data
                    """
                    return {
                        'traj': Transition(
                            **{field: jnp.reshape(getattr(data['traj'], field), 
                                                [config["NUM_MINIBATCHES"], -1] + list(getattr(data['traj'], field).shape[1:]))
                               for field in data['traj']._fields}
                        ),
                        'advantages': data['advantages'].reshape(config["NUM_MINIBATCHES"], -1),
                        'targets': data['targets'].reshape(config["NUM_MINIBATCHES"], -1)
                    }

                minibatches = {
                    agent: create_minibatches(data)
                    for agent, data in agent_data.items()
                }

                print("\nMinibatches structure:", jax.tree_map(lambda x: x.shape if hasattr(x, 'shape') else type(x), minibatches))

                # Update networks using minibatches
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )

                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = info
            current_timestep = update_step*config["NUM_STEPS"]*config["NUM_ENVS"]
            metric["shaped_reward"] = metric["shaped_reward"]["agent_0"]
            metric["shaped_reward_annealed"] = metric["shaped_reward"]*rew_shaping_anneal(current_timestep)
            
            rng = update_state[-1]

            def callback(metric):
                """Log training metrics to wandb.
                
                This function logs the training metrics to wandb, which are used for
                monitoring and analysis during training.

                Args:
                    metric: Training metrics to be logged
                """
                wandb.log(
                    metric
                )
            update_step = update_step + 1
            metric = jax.tree.map(lambda x: x.mean(), metric)
            metric["update_step"] = update_step
            metric["env_step"] = update_step*config["NUM_STEPS"]*config["NUM_ENVS"]
            jax.debug.callback(callback, metric)
            
            runner_state = (train_state, env_state, last_obs, update_step, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train



@hydra.main(version_base=None, config_path="config", config_name="ippo_ff_overcooked_oracle")
def main(config):
    """Main entry point for training"""
    # Process config
    config = OmegaConf.to_container(config) 
    layout_name = config["ENV_KWARGS"]["layout"]
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]

    # Initialize wandb logging
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "FF", "Oracle"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f'ippo_ff_overcooked_{layout_name}'
    )

    # Setup random seeds and training
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])    
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

    # Generate visualization
    filename = f'{config["ENV_NAME"]}_{layout_name}'
    train_state = jax.tree.map(lambda x: x[0], out["runner_state"][0])
    state_seq = get_rollout(train_state, config)
    viz = OvercookedVisualizer()
    # agent_view_size is hardcoded as it determines the padding around the layout.
    viz.animate(state_seq, agent_view_size=5, filename=f"{filename}.gif")
    
    
    """
    print('** Saving Results **')
    filename = f'{config["ENV_NAME"]}_cramped_room_new'
    rewards = out["metrics"]["returned_episode_returns"].mean(-1).reshape((num_seeds, -1))
    reward_mean = rewards.mean(0)  # mean 
    reward_std = rewards.std(0) / np.sqrt(num_seeds)  # standard error
    
    plt.plot(reward_mean)
    plt.fill_between(range(len(reward_mean)), reward_mean - reward_std, reward_mean + reward_std, alpha=0.2)
    # compute standard error
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.savefig(f'{filename}.png')

    # animate first seed
    train_state = jax.tree.map(lambda x: x[0], out["runner_state"][0])
    state_seq = get_rollout(train_state, config)
    viz = OvercookedVisualizer()
    # agent_view_size is hardcoded as it determines the padding around the layout.
    viz.animate(state_seq, agent_view_size=5, filename=f"{filename}.gif")
    """

if __name__ == "__main__":
    main()