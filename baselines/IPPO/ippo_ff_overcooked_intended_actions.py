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
    """Generate a single episode rollout for visualization with intended action augmentation"""
    # Initialize environment
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    # env_params = env.default_params
    # env = LogWrapper(env)

    # Initialize network
    network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
    
    # Initialize random keys
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    # Initialize observation
    init_x = jnp.zeros(env.observation_space().shape)
    init_x = init_x.flatten()
    network.init(key_a, init_x)
    network_params = train_state.params

    # Reset environment and initialize tracking lists
    obs, state = env.reset(key_r)
    state_seq = [state]
    rewards = []
    shaped_rewards = []
    done = False
    
    # Run episode until completion
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        # 1. Get original observations
        obs = {k: v.flatten() for k, v in obs.items()}
        print("Original observation shapes:")
        print("agent_0 obs shape:", obs["agent_0"].shape)
        print("agent_1 obs shape:", obs["agent_1"].shape)

        # 2. Get intended actions using original observations
        pi_0, _ = network.apply(network_params, obs["agent_0"])
        pi_1, _ = network.apply(network_params, obs["agent_1"])
        
        intended_actions = {
            "agent_0": pi_0.sample(seed=key_a0),
            "agent_1": pi_1.sample(seed=key_a1)
        }
        print("Intended actions:", intended_actions)

        # 3. Create augmented observations with co-player's intended actions
        aug_obs = {
            "agent_0": jnp.concatenate([obs["agent_0"], jnp.array([intended_actions["agent_1"]])]),
            "agent_1": jnp.concatenate([obs["agent_1"], jnp.array([intended_actions["agent_0"]])])
        }
        print("Augmented observation shapes:")
        print("agent_0 aug obs shape:", aug_obs["agent_0"].shape)
        print("agent_1 aug obs shape:", aug_obs["agent_1"].shape)

        # 4. Get final actions using augmented observations
        pi_0_aug, _ = network.apply(network_params, aug_obs["agent_0"])
        pi_1_aug, _ = network.apply(network_params, aug_obs["agent_1"])
        
        actions = {
            "agent_0": pi_0_aug.sample(seed=key_a0),
            "agent_1": pi_1_aug.sample(seed=key_a1)
        }
        print("Final actions:", actions)

        # 5. Step environment with final actions
        obs, state, reward, done, info = env.step(key_s, state, actions)
        print("Reward:", reward)
        print("Shaped reward:", info["shaped_reward"])
        
        # Process done condition and store rewards
        done = done["__all__"]
        rewards.append(reward['agent_0'])
        shaped_rewards.append(info["shaped_reward"]['agent_0'])
        state_seq.append(state)

    # Plot rewards for visualization
    plt.figure()
    plt.plot(rewards, label="reward")
    plt.plot(shaped_rewards, label="shaped_reward")
    plt.legend()
    plt.savefig("reward.png")
    plt.show()
    plt.close()

    return state_seq

def batchify(x: dict, agent_list, num_actors):
    """Convert dict of agent observations to batched array"""
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    """Convert batched array back to dict of agent observations"""
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(config):
    """Creates the main training function with the given config"""
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
        # Initialize network and optimizer
        network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space().shape)
        
        init_x = init_x.flatten()
        print("init_x shape:", init_x.shape)
        
        network_params = network.init(_rng, init_x)
        
        # Setup optimizer with optional learning rate annealing
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        
        # Initialize environment states
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        
        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, update_step, rng = runner_state

                # Split RNGs
                rng, _rng = jax.random.split(rng)
                rng, key_a0, key_a1, key_step = jax.random.split(_rng, 4)

                # Debug prints for observation shapes
                print("Initial observation shapes:")
                print(f"agent_0 obs shape: {jax.tree_map(lambda x: x.shape, last_obs['agent_0'])}")
                print(f"agent_1 obs shape: {jax.tree_map(lambda x: x.shape, last_obs['agent_1'])}")

                # 1. Get original batched observations
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])

                print("obs_batch shape:", obs_batch.shape)
                
                pi, value = network.apply(train_state.params, obs_batch)
                intended_actions = pi.sample(seed=_rng)
                print("intended actions shape:", intended_actions.shape)
                intended_actions_dict = unbatchify(intended_actions, env.agents, config["NUM_ENVS"], env.num_agents)
                intended_actions_dict = {k: v.flatten() for k, v in intended_actions_dict.items()}
                print("intended_actions:", intended_actions_dict)

                # 3. Create augmented observations with co-player's intended actions
                aug_obs = {
                    "agent_0": jnp.concatenate([last_obs["agent_0"].flatten(), jnp.array([intended_actions_dict["agent_1"]])]),
                    "agent_1": jnp.concatenate([last_obs["agent_1"].flatten(), jnp.array([intended_actions_dict["agent_0"]])])
                }
                aug_obs_batch = batchify(aug_obs, env.agents, config["NUM_ACTORS"])

                # 4. Compute final actions using augmented observations
                pi_aug, value = network.apply(train_state.params, aug_obs_batch)
                action = pi_aug.sample(seed=_rng)
                print("final action shape:", action.shape)
                log_prob = pi_aug.log_prob(action)

                # 5. Prepare actions for environment step
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}
                print("env_act:", env_act)

                # 6. Step environment
                rng_step = jax.random.split(key_step, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    rng_step, env_state, env_act
                )

                print("reward:", reward)
                print("shaped reward:", info["shaped_reward"])

                # 7. Process rewards and info
                info["reward"] = reward["agent_0"]
                current_timestep = update_step*config["NUM_STEPS"]*config["NUM_ENVS"]
                reward = jax.tree.map(lambda x,y: x+y*rew_shaping_anneal(current_timestep), reward, info["shaped_reward"])

                # 8. Create transition with augmented observations
                transition = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    aug_obs_batch,  # Store augmented observations
                )

                runner_state = (train_state, env_state, obsv, update_step, rng)
                return runner_state, (transition, info)

            runner_state, (traj_batch, info) = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            
            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, update_step, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            _, last_val = network.apply(train_state.params, last_obs_batch)

            def _calculate_gae(traj_batch, last_val):
                """Calculate generalized advantage estimation.
                Note: This function remains unchanged because GAE calculation
                only depends on values, rewards, and done flags."""
                
                def _get_advantages(gae_and_next_value, transition):
                    """Calculate advantages for a single transition.
                    Args:
                        gae_and_next_value: tuple (gae, next_value)
                        transition: Transition object containing (done, value, reward)
                    """
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    # Standard GAE calculation
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                # Calculate advantages for entire trajectory
                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)
            
            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        """Loss function that handles both original and augmented observations"""
                        # Split augmented observation into original obs and intended action
                        orig_obs_size = traj_batch.obs.shape[-1] - 1  # Last dimension is intended action
                        orig_obs = traj_batch.obs[..., :orig_obs_size]
                        intended_actions = traj_batch.obs[..., -1:]

                        # First pass: get intended actions (for consistency with rollout)
                        pi_orig, _ = network.apply(params, orig_obs)
                        
                        # Second pass: use augmented observations for final policy and value
                        pi, value = network.apply(params, traj_batch.obs)  # Full augmented observation
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

                        # Optional: Add consistency loss between original and augmented policies
                        # pi_orig_probs = jax.nn.softmax(pi_orig.logits)
                        # pi_aug_probs = jax.nn.softmax(pi.logits)
                        # consistency_loss = jnp.mean(jnp.square(pi_aug_probs - pi_orig_probs))

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                            # + config.get("CONSISTENCY_COEF", 0.0) * consistency_loss  # Optional
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
        runner_state = (train_state, env_state, obsv, 0, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train



@hydra.main(version_base=None, config_path="config", config_name="ippo_ff_overcooked")
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
        tags=["IPPO", "FF"],
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