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
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
import hydra
from omegaconf import OmegaConf
import wandb

import matplotlib.pyplot as plt

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    agent_type: str = "partner"  # "ego" or "partner"

    @nn.compact
    def __call__(self, x):
        # Set activation function
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Handle ego agent's special input processing
        if self.agent_type == "ego":
            # Calculate state dimension
            state_dim = x.shape[-1] - self.action_dim
            
            # Handle both single and batched inputs
            if x.ndim == 1:
                # Single observation case
                state = x[:state_dim]
                partner_action = x[state_dim:]
                x = jnp.concatenate([state, partner_action])
            else:
                # Batched observation case
                state = x[:, :state_dim]
                partner_action = x[:, state_dim:]
                x = jnp.concatenate([state, partner_action], axis=-1)

        # Actor network
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
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

def get_rollout(train_state, config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # Initialize networks with correct agent types
    partner_network = ActorCritic(
        env.action_space().n,
        activation=config["ACTIVATION"],
        agent_type="partner"
    )
    ego_network = ActorCritic(
        env.action_space().n,
        activation=config["ACTIVATION"],
        agent_type="ego"
    )

    # Initial key splits for reset and action sequence
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    # Reset environment
    obs, state = env.reset(key_r)
    
    # Initialize trajectory storage
    state_seq = [state]
    obs_seq = [obs]
    action_seq = []
    rewards = []
    shaped_rewards = []
    done = False

    while not done:
        # Split keys for partner action, ego action, and environment step
        key_partner, key_ego, key_env = jax.random.split(key_a, 3)
        
        # Flatten observations
        obs = {k: v.flatten() for k, v in obs.items()}

        # 1. Partner agent acts first
        pi_partner, _ = partner_network.apply(train_state.params, obs["agent_1"])
        partner_action = pi_partner.sample(seed=key_partner)

        # 2. Ego agent acts with augmented observation
        ego_obs = obs["agent_0"]
        partner_action_onehot = jax.nn.one_hot(partner_action, env.action_space().n)
        ego_obs_augmented = jnp.concatenate([ego_obs, partner_action_onehot])

        pi_ego, _ = ego_network.apply(train_state.params, ego_obs_augmented)
        ego_action = pi_ego.sample(seed=key_ego)

        # Combine actions
        actions = {
            "agent_0": ego_action,
            "agent_1": partner_action
        }

        # Step environment
        obs, state, reward, done, info = env.step(key_env, state, actions)
        done = done["__all__"]
        
        # Store trajectory information
        state_seq.append(state)
        obs_seq.append(obs)
        action_seq.append(actions)
        rewards.append(reward['agent_0'])
        shaped_rewards.append(info["shaped_reward"]['agent_0'])

        # Update key_a for next iteration
        key_a = key_env

    # Visualization
    from matplotlib import pyplot as plt
    plt.plot(rewards, label="reward")
    plt.plot(shaped_rewards, label="shaped_reward")
    plt.legend()
    plt.savefig("reward.png")
    plt.close()

    return {
        "state_seq": state_seq,
        "obs_seq": obs_seq,
        "action_seq": action_seq,
        "rewards": rewards,
        "shaped_rewards": shaped_rewards,
    }

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"] 
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    
    env = LogWrapper(env, replace_info=False)
    
    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac
    
    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.,
        end_value=0.,
        transition_steps=config["REW_SHAPING_HORIZON"]
    )

    def train(rng):
        # INIT NETWORKS
        partner_network = ActorCritic(
            action_dim=env.action_space().n,
            activation=config["ACTIVATION"],
            agent_type="partner"
        )
        ego_network = ActorCritic(
            action_dim=env.action_space().n,
            activation=config["ACTIVATION"],
            agent_type="ego"
        )

        # Initialize observation space
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space().shape)
        init_x = init_x.flatten()

        # Initialize network parameters
        rng, init_rng_partner, init_rng_ego = jax.random.split(rng, 3)
        partner_params = partner_network.init(init_rng_partner, init_x)
        ego_params = ego_network.init(
            init_rng_ego,
            jnp.concatenate([init_x, jnp.zeros(env.action_space().n)])
        )

        # Setup optimizers
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5)
            )

        # Create train states for both networks
        partner_train_state = TrainState.create(
            apply_fn=partner_network.apply,
            params=partner_params,
            tx=tx,
        )
        ego_train_state = TrainState.create(
            apply_fn=ego_network.apply,
            params=ego_params,
            tx=tx,
        )
        
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        
        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                partner_train_state, ego_train_state, env_state, last_obs, update_step, rng = runner_state

                # SELECT ACTIONS SEQUENTIALLY
                rng, key_partner, key_ego = jax.random.split(rng, 3)

                # 1. Partner agent acts first
                partner_obs = batchify(last_obs, ["agent_1"], config["NUM_ACTORS"])
                pi_partner, value_partner = partner_network.apply(
                    partner_train_state.params,
                    partner_obs
                )
                partner_action = pi_partner.sample(seed=key_partner)
                partner_log_prob = pi_partner.log_prob(partner_action)

                # 2. Ego agent acts with augmented observation
                ego_obs = batchify(last_obs, ["agent_0"], config["NUM_ACTORS"])
                partner_action_onehot = jax.nn.one_hot(partner_action, env.action_space().n)
                ego_obs_augmented = jnp.concatenate([ego_obs, partner_action_onehot], axis=-1)

                pi_ego, value_ego = ego_network.apply(
                    ego_train_state.params,
                    ego_obs_augmented
                )
                ego_action = pi_ego.sample(seed=key_ego)
                ego_log_prob = pi_ego.log_prob(ego_action)

                # Combine actions
                action = jnp.stack([ego_action, partner_action])
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                # STEP ENV
                rng, key_step = jax.random.split(rng)
                rng_step = jax.random.split(key_step, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    rng_step, env_state, env_act
                )

                info["reward"] = reward["agent_0"]
                current_timestep = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                reward = jax.tree_util.tree_map(
                    lambda x, y: x + y * rew_shaping_anneal(current_timestep),
                    reward,
                    info["shaped_reward"]
                )

                # Create transition for both agents
                transition = Transition(
                    done=batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action=action,
                    value=jnp.stack([value_ego, value_partner]),
                    reward=batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob=jnp.stack([ego_log_prob, partner_log_prob]),
                    obs=jnp.stack([ego_obs_augmented, partner_obs])
                )

                runner_state = (partner_train_state, ego_train_state, env_state, obsv, update_step, rng)
                return runner_state, (transition, info)

            runner_state, (traj_batch, info) = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            
            # CALCULATE ADVANTAGE
            partner_train_state, ego_train_state, env_state, last_obs, update_step, rng = runner_state
            
            # Calculate last values for both agents
            partner_obs = batchify(last_obs, ["agent_1"], config["NUM_ACTORS"])
            ego_obs = batchify(last_obs, ["agent_0"], config["NUM_ACTORS"])
            
            # For ego agent, we need partner's last action (using zero action as placeholder)
            ego_obs_augmented = jnp.concatenate([
                ego_obs,
                jnp.zeros((ego_obs.shape[0], env.action_space().n))
            ], axis=-1)
            
            _, last_val_partner = partner_network.apply(partner_train_state.params, partner_obs)
            _, last_val_ego = ego_network.apply(ego_train_state.params, ego_obs_augmented)
            last_val = jnp.stack([last_val_ego, last_val_partner])

            def _calculate_gae(traj_batch, last_val):
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

            advantages, targets = _calculate_gae(traj_batch, last_val)
            
            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_states, batch_info):
                    partner_train_state, ego_train_state = train_states
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, network, traj_batch, gae, targets, agent_idx):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs[agent_idx])
                        log_prob = pi.log_prob(traj_batch.action[agent_idx])

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value[agent_idx] + (
                            value - traj_batch.value[agent_idx]
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets[agent_idx])
                        value_losses_clipped = jnp.square(value_pred_clipped - targets[agent_idx])
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob[agent_idx])
                        gae_agent = (gae[agent_idx] - gae[agent_idx].mean()) / (gae[agent_idx].std() + 1e-8)
                        loss_actor1 = ratio * gae_agent
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae_agent
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

                    # Update partner network
                    grad_fn_partner = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss_partner, grads_partner = grad_fn_partner(
                        partner_train_state.params, partner_network, traj_batch, advantages, targets, 1
                    )
                    partner_train_state = partner_train_state.apply_gradients(grads=grads_partner)

                    # Update ego network
                    grad_fn_ego = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss_ego, grads_ego = grad_fn_ego(
                        ego_train_state.params, ego_network, traj_batch, advantages, targets, 0
                    )
                    ego_train_state = ego_train_state.apply_gradients(grads=grads_ego)

                    return (partner_train_state, ego_train_state), (total_loss_partner, total_loss_ego)

                train_states, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                
                train_states, total_loss = jax.lax.scan(
                    _update_minibatch, train_states, minibatches
                )
                
                update_state = (train_states, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = ((partner_train_state, ego_train_state), traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            
            partner_train_state, ego_train_state = update_state[0]
            metric = info
            current_timestep = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric["shaped_reward"] = metric["shaped_reward"]["agent_0"]
            metric["shaped_reward_annealed"] = metric["shaped_reward"] * rew_shaping_anneal(current_timestep)
            
            rng = update_state[-1]

            def callback(metric):
                wandb.log(metric)
                
            update_step = update_step + 1
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            jax.debug.callback(callback, metric)
            
            runner_state = (partner_train_state, ego_train_state, env_state, obsv, update_step, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (partner_train_state, ego_train_state, env_state, obsv, 0, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}



@hydra.main(version_base=None, config_path="config", config_name="ippo_ff_overcooked")
def main(config):
    config = OmegaConf.to_container(config) 
    layout_name = config["ENV_KWARGS"]["layout"]
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "FF", "RE_ACTION"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f'ippo_ff_overcooked_{layout_name}'
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])    
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

    filename = f'{config["ENV_NAME"]}_{layout_name}'
    train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0])
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
    train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0])
    state_seq = get_rollout(train_state, config)
    viz = OvercookedVisualizer()
    # agent_view_size is hardcoded as it determines the padding around the layout.
    viz.animate(state_seq, agent_view_size=5, filename=f"{filename}.gif")
    """

if __name__ == "__main__":
    main()