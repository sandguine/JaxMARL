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

# Global dimensions
class EnvDimensions:
    """Container for environment dimensions"""
    def __init__(self):
        self.base_obs_shape = None
        self.base_obs_dim = None
        self.action_dim = None
        self.agent_0_obs_dim = None
    
    @classmethod
    def from_env(cls, env):
        dims = cls()
        dims.base_obs_shape = env.observation_space().shape
        dims.base_obs_dim = np.prod(dims.base_obs_shape)
        dims.action_dim = env.action_space().n
        dims.agent_0_obs_dim = dims.base_obs_dim + dims.action_dim
        return dims

# Global dimensions
DIMS = EnvDimensions()

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
    """Generate a single episode rollout for visualization"""
    # Initialize environment
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    # env_params = env.default_params
    # env = LogWrapper(env)
    global DIMS
    DIMS = EnvDimensions.from_env(env)

    # Initialize network
    network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
    
    # Initialize seeds
    key = jax.random.PRNGKey(0)
    key, key_a, key_r = jax.random.split(key, 3)
    # Split key_a for agent_1/agent_0 network init
    key_a_agent_1, key_a_agent_0 = jax.random.split(key_a)
    # key_r for environment reset
    # key for future episode steps

    # agent_1 network init
    init_x_agent_1 = jnp.zeros(DIMS.base_obs_dim)
    init_x_agent_1 = init_x_agent_1.flatten()
    
    network.init(key_a_agent_1, init_x_agent_1)
    network_params_agent_1 = train_state.params['agent_1']

    # agent_0 network init
    init_x_agent_0 = jnp.zeros(DIMS.agent_0_obs_dim)
    init_x_agent_0 = init_x_agent_0.flatten()

    network.init(key_a_agent_0, init_x_agent_0)
    network_params_agent_0 = train_state.params['agent_0']

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

def batchify_agent_1(obs):
    """Convert agent_1 observations to batched array"""
    batched = obs.reshape(obs.shape[0], -1)
    assert batched.shape[1] == DIMS.base_obs_dim, f"Expected shape (-1, {DIMS.base_obs_dim}), got {batched.shape}"
    return batched

def batchify_agent_0(obs, agent_1_action):
    """Convert agent_0 observations to batched array with agent_1 action"""
    batch_size = obs.shape[0]
    base_obs = obs.reshape(batch_size, -1)
    assert base_obs.shape[1] == DIMS.base_obs_dim, f"Expected shape ({batch_size}, {DIMS.base_obs_dim}), got {base_obs.shape}"
    action_oh = jax.nn.one_hot(agent_1_action, DIMS.action_dim)
    return jnp.concatenate([base_obs, action_oh], axis=-1)

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    """Convert batched array back to dict of agent observations"""
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(config):
    """Creates the main training function with the given config"""
    
    # Initialize environment
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    global DIMS
    DIMS = EnvDimensions.from_env(env)

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
        """Learning rate annealing schedule"""
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac
    
    # Schedule for annealing reward shaping
    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.,
        end_value=0.,
        transition_steps=config["REW_SHAPING_HORIZON"]
    )

    def train(rng):
        """Main training loop"""
        # Shapes we're initializing with
        print("Action space:", env.action_space().n)
        print("Observation space shape:", env.observation_space().shape)

        # Initialize network and optimizer
        network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])

        # Initialize seeds
        rng, _rng = jax.random.split(rng)
        _rng_agent_1, _rng_agent_0 = jax.random.split(_rng)  # Split for two networks
        
        # agent_1 network initialization
        init_x_agent_1 = jnp.zeros(DIMS.base_obs_dim)
        init_x_agent_1 = init_x_agent_1.flatten()
        print("init_x_agent_1 shape:", init_x_agent_1.shape)
        
        network_params_agent_1 = network.init(_rng_agent_1, init_x_agent_1)

        # agent_0 network initialization
        init_x_agent_0 = jnp.zeros(DIMS.agent_0_obs_dim)
        init_x_agent_0 = init_x_agent_0.flatten()
        print("init_x_agent_0 shape:", init_x_agent_0.shape)

        network_params_agent_0 = network.init(_rng_agent_0, init_x_agent_0)
        
        # Setup optimizer with optional learning rate annealing
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
        
        # Create combined train state structure
        train_state = TrainState.create(
            apply_fn=network.apply,
            params={
                'agent_1': network_params_agent_1,
                'agent_0': network_params_agent_0
            },
            tx=tx,
        )
        
        # Initialize environment states
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # Process initial observations
        initial_obs = {
            "agent_1": batchify_agent_1(obsv["agent_1"]),
            # For initial step, we can use zeros for agent_1 action since it doesn't exist yet
            "agent_0": batchify_agent_0(obsv["agent_0"], jnp.zeros(config["NUM_ENVS"], dtype=jnp.int32))
        }
        
        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, update_step, rng = runner_state

                # SELECT ACTION consistently with training initialization
                rng, _rng = jax.random.split(rng)
                _rng_agent_1, _rng_agent_0 = jax.random.split(_rng)  # Split for two networks
    

                print("Initial observation shapes:")
                print(f"agent_agent_0 obs shape: {jax.tree_map(lambda x: x.shape, last_obs['agent_0'])}")
                print(f"agent_agent_1 obs shape: {jax.tree_map(lambda x: x.shape, last_obs['agent_1'])}")

                # agent_1 step - using original observation
                print("\nagent_1 shapes:")
                print("Original agent_1 obs shape:", last_obs['agent_1'].shape)
                agent_1_obs_batch = batchify_agent_1(last_obs['agent_1'])
                print("agent_1 obs batch shape:", agent_1_obs_batch.shape)
                agent_1_pi, agent_1_value = network.apply(train_state.params['agent_1'], agent_1_obs_batch)
                agent_1_action = agent_1_pi.sample(seed=_rng_agent_1)
                print("agent_1 action shape:", agent_1_action.shape)
                agent_1_log_prob = agent_1_pi.log_prob(agent_1_action)
                print("agent_1 log prob shape:", agent_1_log_prob.shape)

                # agent_0 step - augmenting observation with agent_1 action
                print("\nagent_0 shapes:")
                print("Original agent_0 obs shape:", last_obs['agent_0'].shape)
                agent_0_obs_batch = batchify_agent_0(last_obs['agent_0'], agent_1_action)
                print("agent_0_obs_batch shape:", agent_0_obs_batch.shape)
                agent_0_pi, agent_0_value = network.apply(train_state.params['agent_0'], agent_0_obs_batch)
                agent_0_action = agent_0_pi.sample(seed=_rng_agent_0)
                print("agent_0_action_shape:", agent_0_action.shape)
                agent_0_log_prob = agent_0_pi.log_prob(agent_0_action)
                print("agent_0_log_prob_shape:", agent_0_log_prob.shape)

                # Package actions and log probabilities for environment step
                log_prob = {
                    'agent_0': agent_0_log_prob,
                    'agent_1': agent_1_log_prob
                }
                env_act = {
                    "agent_0": agent_0_action,
                    "agent_1": agent_1_action
                }
                env_act = {k:v.flatten() for k,v in env_act.items()}
                
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    rng_step, env_state, env_act
                )

                # Process observations
                processed_obs = {
                    "agent_0": batchify_agent_0(obsv["agent_0"], agent_1_action),
                    "agent_1": batchify_agent_1(obsv["agent_1"])
                }

                # Store original reward for logging
                info["reward"] = reward["agent_0"]

                # Apply reward shaping
                current_timestep = update_step*config["NUM_STEPS"]*config["NUM_ENVS"]
                reward = jax.tree.map(
                    lambda x, y: x + y * rew_shaping_anneal(current_timestep),
                    reward,
                    info["shaped_reward"]
                )
                print("shaped reward:", info["shaped_reward"])

                transition = Transition(
                    done=jnp.array([done["agent_0"], done["agent_1"]]).squeeze(),
                    action=jnp.array([agent_0_action, agent_1_action]),
                    value=jnp.array([agent_0_value, agent_1_value]),
                    reward=jnp.array([reward["agent_0"], reward["agent_1"]]).squeeze(),
                    log_prob=jnp.array([agent_0_log_prob, agent_1_log_prob]),
                    obs=processed_obs
                )

                runner_state = (train_state, env_state, processed_obs, update_step, rng)
                return runner_state, (transition, info, processed_obs)

            runner_state, (traj_batch, info, processed_obs) = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            
            # CALCULATE ADVANTAGE
            train_state, env_state, _, update_step, rng = runner_state
            
            # Get last values for both agents using final_obs
            _, agent_1_last_val = network.apply(
                train_state.params['agent_1'], 
                processed_obs["agent_1"]
            )
            _, agent_0_last_val = network.apply(
                train_state.params['agent_0'], 
                processed_obs["agent_0"]
            )

            def _calculate_gae_per_agent(traj_batch, last_val):
                """Calculate GAE for a single agent"""
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            # Create separate trajectory batches for each agent
            agent_1_traj = Transition(
                done=traj_batch.done[1],          # agent_1 index
                action=traj_batch.action[1],
                value=traj_batch.value[1],
                reward=traj_batch.reward[1],
                log_prob=traj_batch.log_prob[1],
                obs=traj_batch.obs["agent_1"]
            )

            agent_0_traj = Transition(
                done=traj_batch.done[0],          # agent_0 index
                action=traj_batch.action[0],
                value=traj_batch.value[0],
                reward=traj_batch.reward[0],
                log_prob=traj_batch.log_prob[0],
                obs=traj_batch.obs["agent_0"]
            )

            # Calculate advantages separately for each agent
            agent_1_advantages, agent_1_targets = _calculate_gae_per_agent(
                agent_1_traj, 
                agent_1_last_val
            )
            
            agent_0_advantages, agent_0_targets = _calculate_gae_per_agent(
                agent_0_traj, 
                agent_0_last_val
            )

            # Combine advantages and targets
            advantages = jnp.array([agent_0_advantages, agent_1_advantages])
            targets = jnp.array([agent_0_targets, agent_1_targets])
            
            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
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

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
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
        runner_state = (train_state, env_state, initial_obs, 0, _rng)
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