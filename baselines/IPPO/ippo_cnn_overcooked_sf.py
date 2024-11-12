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
import copy

from jaxmarl.next_action_predictor import ActionPredictor, create_train_state, predict_action

class CNN(nn.Module):
    activation: str = "tanh"
    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        x = nn.Conv(features=32, kernel_size=(5, 5), kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        return x

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        embedding = CNN(self.activation)(x)
        actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)
        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return pi, jnp.squeeze(critic, axis=-1)

class SuccessorFeatureNetwork(nn.Module):
    num_features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_features)(x)
        return x

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def get_rollout(params, config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)
    done = False
    obs, state = env.reset(key_r)
    state_seq = [state]
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)
        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space().shape)
        pi, value = network.apply(params, obs_batch)
        action = pi.sample(seed=key_a0)
        env_act = unbatchify(action, env.agents, 1, env.num_agents)
        env_act = {k: v.squeeze() for k, v in env_act.items()}
        obs, state, reward, done, info = env.step(key_s, state, env_act)
        done = done["__all__"]
        state_seq.append(state)
    return state_seq

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(config, use_successor_features=False):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    env = LogWrapper(env, replace_info=False)

    # Initialize networks
    network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
    rng, _rng = jax.random.split(jax.random.PRNGKey(config["SEED"]))
    init_x = jnp.zeros((1, *env.observation_space().shape))
    network_params = network.init(_rng, init_x)
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    if use_successor_features:
        sf_network = SuccessorFeatureNetwork(num_features=env.observation_space().shape[0])
        sf_key = jax.random.PRNGKey(config["SEED"])
        sf_params = sf_network.init(sf_key, init_x)

    def train(rng):
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, update_step, rng = runner_state
            current_state = jnp.stack([last_obs[a] for a in env.agents]).reshape(-1, *env.observation_space().shape)

            if use_successor_features:
                sf_current = sf_network.apply(sf_params, current_state)
                combined_features = jnp.concatenate([current_state, sf_current], axis=-1)
            else:
                combined_features = current_state

            rng, _rng = jax.random.split(rng)
            pi, value = network.apply(train_state.params, combined_features)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
            env_act = {k: v.flatten() for k, v in env_act.items()}

            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(rng_step, env_state, env_act)

            transition = Transition(
                batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                action,
                value,
                batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                log_prob,
                combined_features,
                info,
            )

            runner_state = (train_state, env_state, obsv, update_step, rng)
            return runner_state, transition

        runner_state = (train_state, env_state, obsv, 0, rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

def single_run(config, use_successor_features=False):
    config = OmegaConf.to_container(config)
    layout_name = copy.deepcopy(config["ENV_KWARGS"]["layout"])
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]

    run_name = f'ippo_cnn_overcooked_{"sf" if use_successor_features else "baseline"}_{layout_name}'
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "FF", "SuccessorFeatures" if use_successor_features else "Baseline"],
        config=config,
        mode=config["WANDB_MODE"],
        name=run_name
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config, use_successor_features))
    out = jax.vmap(train_jit)(rngs)

    print("** Saving Results **")
    filename = f'{config["ENV_NAME"]}_{layout_name}_seed{config["SEED"]}'
    train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0])
    state_seq = get_rollout(train_state.params, config)
    viz = OvercookedVisualizer()
    viz.animate(state_seq, agent_view_size=5, filename=f"{filename}.gif")

def tune(default_config):
    default_config = OmegaConf.to_container(default_config)
    layout_name = default_config["ENV_KWARGS"]["layout"]

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])
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

@hydra.main(version_base=None, config_path="config", config_name="ippo_cnn_overcooked")
def main(config):
    if config["TUNE"]:
        tune(config)
    else:
        single_run(config)

if __name__ == "__main__":
    main()
