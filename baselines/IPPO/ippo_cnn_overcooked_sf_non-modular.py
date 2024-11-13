"""
IPPO Implementation with Successor Features option for Overcooked
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Optional
from flax.training.train_state import TrainState
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
import hydra
from omegaconf import OmegaConf
import wandb
import copy

# Network Architectures
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
    feature_dim: Optional[int] = None  # Add feature dimension for SF

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        
        # Handle both standard input and SF-augmented input
        if self.feature_dim is not None:
            # For SF case, x is already the concatenated features
            embedding = x
        else:
            # Standard case - process raw observation
            embedding = CNN(self.activation)(x)
            
        actor_mean = nn.Dense(
            64, 
            kernel_init=orthogonal(np.sqrt(2)), 
            bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, 
            kernel_init=orthogonal(0.01), 
            bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)
        
        critic = nn.Dense(
            64, 
            kernel_init=orthogonal(np.sqrt(2)), 
            bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(
            1, 
            kernel_init=orthogonal(1.0), 
            bias_init=constant(0.0)
        )(critic)
        
        return pi, jnp.squeeze(critic, axis=-1)

class FeatureEncoder(nn.Module):
    num_features: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        x = CNN(self.activation)(x)
        x = nn.Dense(
            self.num_features, 
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        return x

class SuccessorFeatureNetwork(nn.Module):
    num_features: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        x = nn.Dense(
            128, 
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = activation(x)
        x = nn.Dense(
            self.num_features, 
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        
        return x

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: Any

class TrainingState(NamedTuple):
    policy_state: TrainState
    feature_state: TrainState
    sf_state: TrainState
    env_state: Any
    last_obs: jnp.ndarray
    rng: Any

def make_train_state(rng, config, env):
    """Initialize all training states"""
    rng, _rng = jax.random.split(rng)
    
    if config["USE_SF"]:
        sr_wrapper = SRWrapper(config)
        actor_critic = SRActorCritic(
            env.action_space().n,
            use_sf=True,
            feature_dim=config["FEATURE_DIM"]
        )
    else:
        actor_critic = SRActorCritic(env.action_space().n)
    
    # Initialize dummy inputs
    dummy_obs = jnp.zeros((1, *env.observation_space().shape))

    # Initialize parameters with proper random keys
    rng, key_ac, key_fe, key_sf = jax.random.split(rng, 4)
    
    if config["USE_SF"]:
        dummy_features = jnp.zeros((1, config["FEATURE_DIM"] * 2))
        actor_critic_params = actor_critic.init(key_ac, dummy_features)
    else:
        actor_critic_params = actor_critic.init(key_ac, dummy_obs)
        
    feature_params = feature_encoder.init(key_fe, dummy_obs)
    sf_params = sf_network.init(key_sf, jnp.zeros((1, config["FEATURE_DIM"])))

    # Create optimizers
    policy_tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5)
    )
    feature_tx = optax.adam(config["FEATURE_LR"])
    sf_tx = optax.adam(config["SF_LR"])

    # Create train states
    policy_state = TrainState.create(
        apply_fn=actor_critic.apply,
        params=actor_critic_params,
        tx=policy_tx,
    )
    
    feature_state = TrainState.create(
        apply_fn=feature_encoder.apply,
        params=feature_params,
        tx=feature_tx,
    )
    
    sf_state = TrainState.create(
        apply_fn=sf_network.apply,
        params=sf_params,
        tx=sf_tx,
    )

    return policy_state, feature_state, sf_state

def compute_gae(rewards, values, dones, last_value, config):
    """Compute generalized advantage estimation"""
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        done, value, reward = transition
        
        delta = reward + config["GAMMA"] * next_value * (1 - done) - value
        gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
        
        return (gae, value), gae
    
    values_t = jnp.concatenate([values, last_value[None, :]], axis=0)
    advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_value), values_t[-1]),
        (dones, values, rewards),
        reverse=True
    )[1]
    
    targets = advantages + values
    
    return advantages, targets

def collect_trajectories(train_state, feature_state, sf_state, env_state, last_obs, rng, config):
    """Collect trajectories using current policy"""
    def _env_step(runner_state, unused):
        train_state, feature_state, sf_state, env_state, last_obs, rng = runner_state
        
        rng, _rng = jax.random.split(rng)
        
        if config["USE_SF"]:
            sr_wrapper = SRWrapper(config)
            encoded_state, sf_features = sr_wrapper.process_observation(
                last_obs,
                train_state
            )
            pi, value = train_state.apply_fn(train_state.params, encoded_state)
        else:
            pi, value = train_state.apply_fn(train_state.params, last_obs)
        
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)
        
        rng, _rng = jax.random.split(rng)
        next_obs, next_env_state, reward, done, info = env.step(_rng, env_state, action)
        
        transition = Transition(
            done=done,
            action=action,
            value=value,
            reward=reward,
            log_prob=log_prob,
            obs=last_obs,
            next_obs=next_obs,
            info=info
        )
        
        next_runner_state = (
            train_state,
            feature_state,
            sf_state,
            next_env_state,
            next_obs,
            rng
        )
        
        return next_runner_state, transition
    
    # Collect trajectories
    runner_state = (train_state, feature_state, sf_state, env_state, last_obs, rng)
    runner_state, traj_batch = jax.lax.scan(
        _env_step,
        runner_state,
        None,
        config["NUM_STEPS"]
    )
    
    return runner_state, traj_batch

def _update_minibatch(train_state, feature_state, sf_state, batch, config):
    """Update policy parameters on a minibatch"""
    traj_batch, advantages, targets = batch
    
    def _loss_fn(policy_params, feature_params=None, sf_params=None):
        if config["USE_SF"]:
            # Get encoded features and successor features
            encoded_state = feature_state.apply_fn(
                feature_params, 
                traj_batch.obs
            )
            successor_features = sf_state.apply_fn(
                sf_params, 
                encoded_state
            )
            combined_features = jnp.concatenate([encoded_state, successor_features], axis=-1)
            pi, value = train_state.apply_fn(policy_params, combined_features)
            
            # Compute SF prediction loss
            next_encoded_state = feature_state.apply_fn(
                feature_params, 
                traj_batch.next_obs
            )
            next_successor_features = sf_state.apply_fn(
                sf_params, 
                next_encoded_state
            )
            sf_target = encoded_state + config["GAMMA"] * (1 - traj_batch.done) * next_successor_features
            sf_loss = config["SF_COEF"] * optax.huber_loss(successor_features, sf_target).mean()
        else:
            pi, value = train_state.apply_fn(policy_params, traj_batch.obs)
            sf_loss = 0.0
        
        # Policy loss
        log_prob = pi.log_prob(traj_batch.action)
        ratio = jnp.exp(log_prob - traj_batch.log_prob)
        clip_ratio = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"])
        policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clip_ratio * advantages))
        
        # Value loss
        value_loss = config["VF_COEF"] * jnp.mean((value - targets) ** 2)
        
        # Entropy loss
        entropy = jnp.mean(pi.entropy())
        entropy_loss = -config["ENT_COEF"] * entropy
        
        total_loss = policy_loss + value_loss + entropy_loss + sf_loss
        
        return total_loss, {
            "total_loss": total_loss,
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "entropy": entropy,
            "sf_loss": sf_loss
        }
    
    # Get gradients
    if config["USE_SF"]:
        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True, argnums=(0, 1, 2))
        (loss, metrics), grads = grad_fn(
            train_state.params,
            feature_state.params,
            sf_state.params
        )
        # Update all networks
        new_train_state = train_state.apply_gradients(grads=grads[0])
        new_feature_state = feature_state.apply_gradients(grads=grads[1])
        new_sf_state = sf_state.apply_gradients(grads=grads[2])
    else:
        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(train_state.params)
        # Update only policy network
        new_train_state = train_state.apply_gradients(grads=grads)
        new_feature_state = feature_state
        new_sf_state = sf_state
    
    return (new_train_state, new_feature_state, new_sf_state), metrics

def make_train(config):
    """Creates training update step"""
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    
    def _update_step(runner_state, unused):
        train_state, feature_state, sf_state, env_state, last_obs, rng = runner_state
        
        # Collect trajectories
        rng, _rng = jax.random.split(rng)
        runner_state, traj_batch = collect_trajectories(
            train_state, feature_state, sf_state, env_state, last_obs, _rng, config
        )

        # Compute advantages
        if config["USE_SF"]:
            encoded_state = feature_state.apply_fn(
                feature_state.params,
                runner_state[4]  # last_obs
            )
            successor_features = sf_state.apply_fn(
                sf_state.params,
                encoded_state
            )
            combined_features = jnp.concatenate([encoded_state, successor_features], axis=-1)
            _, last_val = train_state.apply_fn(train_state.params, combined_features)
        else:
            _, last_val = train_state.apply_fn(train_state.params, runner_state[4])

        advantages, targets = compute_gae(
            traj_batch.reward,
            traj_batch.value,
            traj_batch.done,
            last_val,
            config
        )

        # Update networks
        rng, _rng = jax.random.split(rng)
        (new_train_state, new_feature_state, new_sf_state), metrics = _update_minibatch(
            train_state,
            feature_state,
            sf_state,
            (traj_batch, advantages, targets),
            config
        )

        metrics["returned_episode_returns"] = jnp.mean(traj_batch.info["returned_episode_returns"])
        metrics["returned_episode_lengths"] = jnp.mean(traj_batch.info["returned_episode_lengths"])
        
        runner_state = (new_train_state, new_feature_state, new_sf_state, env_state, last_obs, rng)
        return runner_state, metrics

    return _update_step

def train(config):
    """Main training loop"""
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    
    # Calculate number of updates based on total timesteps
    num_updates = int(config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"] * config["NUM_ENVS"]))
    
    # Initialize random key
    rng = jax.random.PRNGKey(config["SEED"])
    
    # Initialize training states
    rng, _rng = jax.random.split(rng)
    train_state, feature_state, sf_state = make_train_state(_rng, config, env)
    
    # Initialize environment
    rng, _rng = jax.random.split(rng)
    obsv, env_state = env.reset(_rng)
    
    # Create initial runner state
    runner_state = (train_state, feature_state, sf_state, env_state, obsv, rng)
    
    # Create training update function
    update_step = make_train(config)
    
    # Training loop
    for update in range(config["NUM_UPDATES"]):
        runner_state, metrics = update_step(runner_state, None)
        
        # Logging
        if (update + 1) % config["LOG_INTERVAL"] == 0:
            metrics = jax.tree_util.tree_map(lambda x: x.mean(), metrics)
            metrics.update({
                "update": update,
                "step": (update + 1) * config["NUM_STEPS"] * config["NUM_ENVS"],
            })
            wandb.log(metrics)
    
    return runner_state

def evaluate(env, training_state, config, num_episodes=10):
    """Evaluation function"""
    eval_metrics = []
    
    for _ in range(num_episodes):
        rng = jax.random.PRNGKey(config["SEED"])
        obs, env_state = env.reset(rng)
        done = False
        episode_reward = 0
        
        while not done:
            if config["USE_SF"]:
                encoded_state = training_state.feature_state.apply_fn(
                    training_state.feature_state.params,
                    obs
                )
                successor_features = training_state.sf_state.apply_fn(
                    training_state.sf_state.params,
                    encoded_state
                )
                combined_features = jnp.concatenate([encoded_state, successor_features], axis=-1)
                pi, _ = training_state.policy_state.apply_fn(
                    training_state.policy_state.params,
                    combined_features
                )
            else:
                pi, _ = training_state.policy_state.apply_fn(
                    training_state.policy_state.params,
                    obs
                )
            
            action = pi.mode()
            obs, env_state, reward, done, info = env.step(rng, env_state, action)
            episode_reward += reward
        
        eval_metrics.append({"episode_reward": episode_reward})
    
    return eval_metrics

def unbatchify(batch, agents, batch_size, num_agents):
    """Utility function to unbatchify actions"""
    batch = batch.reshape(batch_size, num_agents, -1)
    return {agent: batch[:, i] for i, agent in enumerate(agents)}

def single_run(config):
    # Convert config to dict properly
    config = OmegaConf.to_container(config, resolve=True)
    layout = config["ENV_KWARGS"]["layout"]
    
    # Create a copy of ENV_KWARGS to modify
    env_kwargs = dict(config["ENV_KWARGS"])
    env_kwargs["layout"] = overcooked_layouts[layout]
    config["ENV_KWARGS"] = env_kwargs

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "SF"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f'ippo_cnn_overcooked_sf_{layout}'
    )

    training_state = train(config)

    print("** Saving Results **")
    filename = f'{config["ENV_NAME"]}_{layout}_seed{config["SEED"]}'
    
    # Pass full training state if using SF
    if config["USE_SF"]:
        state_seq = get_rollout(training_state, config)
    else:
        state_seq = get_rollout(training_state.policy_state.params, config)
        
    viz = OvercookedVisualizer()
    viz.animate(state_seq, agent_view_size=5, filename=f"{filename}.gif")

def get_rollout(params, config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    network = ActorCritic(
        env.action_space().n, 
        activation=config["ACTIVATION"],
        feature_dim=config["FEATURE_DIM"] if config["USE_SF"] else None
    )
    
    key = jax.random.PRNGKey(0)
    key, key_reset = jax.random.split(key)
    
    done = False
    obs, state = env.reset(key_reset)
    state_seq = [state]
    
    while not done:
        key, key_act, key_step = jax.random.split(key, 3)
        obs_batch = jnp.stack([obs[a] for a in env.agents])
        obs_batch = obs_batch.reshape(-1, *env.observation_space().shape)
        
        if config["USE_SF"]:
            # Handle SF case
            feature_encoder = FeatureEncoder(config["FEATURE_DIM"], activation=config["ACTIVATION"])
            sf_network = SuccessorFeatureNetwork(config["FEATURE_DIM"], activation=config["ACTIVATION"])
            
            encoded_state = feature_encoder.apply(params.feature_state.params, obs_batch)
            successor_features = sf_network.apply(params.sf_state.params, encoded_state)
            combined_features = jnp.concatenate([encoded_state, successor_features], axis=-1)
            
            pi, value = network.apply(params.policy_state.params, combined_features)
        else:
            pi, value = network.apply(params, obs_batch)
        
        action = pi.sample(seed=key_act)
        env_act = unbatchify(action, env.agents, 1, env.num_agents)
        env_act = {k: v.squeeze() for k, v in env_act.items()}

        obs, state, reward, done, info = env.step(key_step, state, env_act)
        done = done["__all__"]
        state_seq.append(state)

    return state_seq

@hydra.main(version_base=None, config_path="config", config_name="ippo_cnn_overcooked_sf")
def main(config):
    if config["TUNE"]:
        tune(config)
    else:
        single_run(config)

if __name__ == "__main__":
    main()