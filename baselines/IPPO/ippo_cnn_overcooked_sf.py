"""
IPPO Implementation with modular Successor Features for Overcooked
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Optional, Union
from flax.training.train_state import TrainState
import distrax
import wandb
import hydra
from omegaconf import OmegaConf
import copy

import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from jaxmarl.networks.successor_features import FeatureEncoder, SuccessorFeatureNetwork
from jaxmarl.networks.policy import SRActorCritic
from jaxmarl.wrappers.successor_features import SRWrapper

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
    feature_state: Optional[TrainState]
    sf_state: Optional[TrainState]
    env_state: Any
    last_obs: jnp.ndarray
    rng: Any

def make_train_state(rng, config, env):
    """Initialize all training states with proper validation and RNG management"""
    # Validate config
    if config["USE_SF"]:
        required_params = ["FEATURE_DIM", "FEATURE_LR", "SF_LR", "SF_COEF"]
        missing = [p for p in required_params if p not in config]
        if missing:
            raise ValueError(f"Missing required SF parameters: {missing}")

    # Pre-split RNG keys for all components
    rngs = jax.random.split(rng, 4)
    policy_rng, feature_rng, sf_rng, env_rng = rngs
    
    # Initialize networks
    if config["USE_SF"]:
        actor_critic = SRActorCritic(
            action_dim=env.action_space().n,
            use_sf=True,
            feature_dim=config["FEATURE_DIM"],
            activation=config["ACTIVATION"]
        )
        feature_encoder = FeatureEncoder(
            num_features=config["FEATURE_DIM"],
            encoder_type="cnn",
            activation=config["ACTIVATION"]
        )
        sf_network = SuccessorFeatureNetwork(
            num_features=config["FEATURE_DIM"],
            activation=config["ACTIVATION"]
        )
    else:
        actor_critic = SRActorCritic(
            action_dim=env.action_space().n,
            activation=config["ACTIVATION"]
        )
        feature_encoder = None
        sf_network = None

    # Create dummy inputs
    dummy_obs = env.observation_space().sample(rng=rng)
    dummy_obs = dummy_obs.reshape(-1, *env.observation_space().shape)

    # Initialize parameters with specific RNG keys
    if config["USE_SF"]:
        dummy_sf = jnp.zeros((dummy_obs.shape[0], config["FEATURE_DIM"]))
        policy_params = actor_critic.init(policy_rng, dummy_obs, dummy_sf)
        feature_params = feature_encoder.init(feature_rng, dummy_obs)
        dummy_features = jnp.zeros((dummy_obs.shape[0], config["FEATURE_DIM"]))
        sf_params = sf_network.init(sf_rng, dummy_features)
    else:
        policy_params = actor_critic.init(policy_rng, dummy_obs)
        
    # Create train states
    policy_tx = optax.adam(learning_rate=config["LR"])
    policy_state = TrainState.create(
        apply_fn=actor_critic.apply,
        params=policy_params,
        tx=policy_tx,
    )

    if config["USE_SF"]:
        feature_tx = optax.adam(learning_rate=config["FEATURE_LR"])
        feature_state = TrainState.create(
            apply_fn=feature_encoder.apply,
            params=feature_params,
            tx=feature_tx,
        )

        sf_tx = optax.adam(learning_rate=config["SF_LR"])
        sf_state = TrainState.create(
            apply_fn=sf_network.apply,
            params=sf_params,
            tx=sf_tx,
        )
    else:
        feature_state = None
        sf_state = None

    # Initialize environment state and observation
    last_obs, env_state = env.reset(env_rng)

    return TrainingState(
        policy_state=policy_state,
        feature_state=feature_state,
        sf_state=sf_state,
        env_state=env_state,
        last_obs=last_obs,
        rng=rng
    )

def collect_trajectories(train_state, feature_state, sf_state, env_state, last_obs, rng, config):
    """Collect trajectories using current policy"""
    # Validate states
    if config["USE_SF"] and (feature_state is None or sf_state is None):
        raise ValueError("Feature or SF state is None but USE_SF is True")

    def _env_step(runner_state, unused):
        train_state, feature_state, sf_state, env_state, last_obs, rng = runner_state
        
        rng, _rng = jax.random.split(rng)
        
        # Handle batched observations
        if len(last_obs.shape) == 3:  # If missing batch dimension
            last_obs = last_obs[None, ...]
        
        if config["USE_SF"]:
            encoded_state = feature_state.apply_fn(
                feature_state.params,
                last_obs
            )
            sf_features = sf_state.apply_fn(
                sf_state.params,
                encoded_state
            )
            pi, value = train_state.apply_fn(
                train_state.params,
                last_obs,
                sf_features
            )
        else:
            pi, value = train_state.apply_fn(
                train_state.params,
                last_obs
            )
        
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)
        
        # Handle batched environment step
        rng, _rng = jax.random.split(rng)
        next_obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(None, 0, 0))(
            _rng, env_state, action
        )
        
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
        
        runner_state = (train_state, feature_state, sf_state, env_state, next_obs, rng)
        return runner_state, transition
    
    runner_state = (train_state, feature_state, sf_state, env_state, last_obs, rng)
    runner_state, traj_batch = jax.lax.scan(
        _env_step, runner_state, None, config["NUM_STEPS"]
    )
    
    return runner_state, traj_batch

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

def _loss_fn(policy_params, feature_params, sf_params, traj_batch, advantages, targets, config):
    """Compute loss for policy, feature encoder and SF networks"""
    if config["USE_SF"]:
        # Encode states
        encoded_state = feature_state.apply_fn(
            feature_params,
            traj_batch.obs
        )
        next_encoded_state = feature_state.apply_fn(
            feature_params,
            traj_batch.next_obs
        )
        
        # Get SF predictions
        sf_features = sf_state.apply_fn(
            sf_params,
            encoded_state
        )
        
        # Get policy outputs
        pi, value = train_state.apply_fn(
            policy_params,
            traj_batch.obs,
            sf_features
        )
    else:
        pi, value = train_state.apply_fn(
            policy_params,
            traj_batch.obs
        )
    
    # Policy loss
    log_prob = pi.log_prob(traj_batch.action)
    ratio = jnp.exp(log_prob - traj_batch.log_prob)
    
    clip_eps = config["CLIP_EPS"]
    loss1 = ratio * advantages
    loss2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -jnp.minimum(loss1, loss2).mean()
    
    # Value loss
    value_loss = 0.5 * ((value - targets) ** 2).mean()
    
    # Entropy loss for exploration
    entropy_loss = -config["ENT_COEF"] * pi.entropy().mean()
    
    total_loss = policy_loss + value_loss + entropy_loss
    
    metrics = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss
    }
    
    if config["USE_SF"]:
        # SF loss
        sf_loss = compute_sf_loss(encoded_state, next_encoded_state, traj_batch.done, config["GAMMA"])
        total_loss += config["SF_COEF"] * sf_loss
        metrics["sf_loss"] = sf_loss
        
    return total_loss, metrics

def _update_minibatch(train_state, feature_state, sf_state, batch, config, rng):
    """Update networks on minibatches"""
    traj_batch, advantages, targets = batch
    
    # Process in minibatches
    batch_size = traj_batch.obs.shape[0]
    indices = jnp.arange(batch_size)
    indices = jax.random.permutation(rng, indices)
    minibatch_size = batch_size // config["NUM_MINIBATCHES"]
    
    metrics_list = []
    for start in range(0, batch_size, minibatch_size):
        end = start + minibatch_size
        mb_idx = indices[start:end]
        
        # Create minibatch
        mb_traj = jax.tree_map(lambda x: x[mb_idx], traj_batch)
        mb_advantages = advantages[mb_idx]
        mb_targets = targets[mb_idx]
        
        # Compute loss and gradients
        if config["USE_SF"]:
            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True, argnums=(0, 1, 2))
            (loss, mb_metrics), grads = grad_fn(
                train_state.params,
                feature_state.params,
                sf_state.params,
                mb_traj,
                mb_advantages,
                mb_targets,
                config
            )
            # Apply updates
            train_state = train_state.apply_gradients(grads=grads[0])
            feature_state = feature_state.apply_gradients(grads=grads[1])
            sf_state = sf_state.apply_gradients(grads=grads[2])
        else:
            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
            (loss, mb_metrics), grads = grad_fn(
                train_state.params,
                None,
                None,
                mb_traj,
                mb_advantages,
                mb_targets,
                config
            )
            train_state = train_state.apply_gradients(grads=grads)
            
        metrics_list.append(mb_metrics)
    
    # Average metrics across minibatches
    metrics = jax.tree_map(lambda *xs: jnp.mean(jnp.stack(xs)), *metrics_list)
    return train_state, feature_state, sf_state, metrics

def make_train(config):
    """Create training update function"""
    def _update_step(runner_state, unused):
        # Unpack runner state using named variables
        train_state, feature_state, sf_state, env_state, last_obs, rng = runner_state
        
        # Collect trajectories
        runner_state, traj_batch = collect_trajectories(
            train_state, feature_state, sf_state, env_state, last_obs, rng, config
        )
        
        # Unpack new runner state
        train_state, feature_state, sf_state, env_state, last_obs, rng = runner_state
        
        # Compute advantages
        if config["USE_SF"]:
            encoded_state = feature_state.apply_fn(
                feature_state.params,
                runner_state[4]  # last_obs
            )
            sf_features = sf_state.apply_fn(
                sf_state.params,
                encoded_state
            )
            _, last_val = train_state.apply_fn(
                train_state.params,
                runner_state[4],
                sf_features
            )
        else:
            _, last_val = train_state.apply_fn(
                train_state.params,
                runner_state[4]
            )
        
        advantages, targets = compute_gae(
            traj_batch.reward,
            traj_batch.value,
            traj_batch.done,
            last_val,
            config
        )

        # Update networks multiple times
        metrics_list = []
        for _ in range(config["UPDATE_EPOCHS"]):
            train_state, feature_state, sf_state, metrics = _update_minibatch(
                runner_state[0],
                runner_state[1],
                runner_state[2],
                (traj_batch, advantages, targets),
                config,
                rng
            )
            metrics_list.append(metrics)
        
        # Average metrics across epochs
        metrics = jax.tree_map(lambda *xs: jnp.mean(jnp.stack(xs)), *metrics_list)
        
        metrics["returned_episode_returns"] = jnp.mean(traj_batch.info["returned_episode_returns"])
        metrics["returned_episode_lengths"] = jnp.mean(traj_batch.info["returned_episode_lengths"])
        
        return (train_state, feature_state, sf_state, env_state, last_obs, rng), metrics

    return _update_step

def train(config):
    """Main training loop"""
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    if config["USE_SF"]:
        env = SRWrapper(env)
    env = LogWrapper(env)
    
    # Initialize random key
    rng = jax.random.PRNGKey(config["SEED"])
    rng, _rng = jax.random.split(rng)
    
    # Initialize training state
    training_state = make_train_state(_rng, config, env)
    
    # Create runner state tuple
    runner_state = (
        training_state.policy_state,
        training_state.feature_state,
        training_state.sf_state,
        training_state.env_state,
        training_state.last_obs,
        training_state.rng
    )
    
    # Create training update function
    update_step = make_train(config)
    
    # Training loop
    num_updates = config["NUM_UPDATES"]
    for update in range(num_updates):
        runner_state, metrics = update_step(runner_state, None)
        
        if (update + 1) % config["LOG_INTERVAL"] == 0:
            metrics = jax.tree_util.tree_map(lambda x: x.mean(), metrics)
            metrics.update({
                "update": update,
                "step": (update + 1) * config["NUM_STEPS"] * config["NUM_ENVS"],
            })
            wandb.log(metrics)
    
    return runner_state

def single_run(config):
    """Single training run with fixed hyperparameters"""
    config = OmegaConf.to_container(config)
    layout_name = config["ENV_KWARGS"]["layout"]
    env_kwargs = dict(config["ENV_KWARGS"])
    env_kwargs["layout"] = overcooked_layouts[layout_name]
    config["ENV_KWARGS"] = env_kwargs

    # Initialize environment
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = LogWrapper(env)

    # Initialize RNG
    rng = jax.random.PRNGKey(config["SEED"])
    rng, _rng = jax.random.split(rng)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "SF"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f'ippo_cnn_overcooked_sf_{layout_name}'
    )

    training_state = make_train_state(_rng, config, env)
    runner_state = (
        training_state.policy_state,
        training_state.feature_state,
        training_state.sf_state,
        training_state.env_state,
        training_state.last_obs,
        training_state.rng
    )

    print("** Saving Results **")
    filename = f'{config["ENV_NAME"]}_{layout_name}_seed{config["SEED"]}'
    
    # Pass full training state if using SF
    if config["USE_SF"]:
        state_seq = get_rollout(training_state, config)
    else:
        state_seq = get_rollout(training_state.policy_state.params, config)
        
    viz = OvercookedVisualizer()
    viz.animate(state_seq, agent_view_size=5, filename=f"{filename}.gif")

def tune(config):
    """Hyperparameter tuning with wandb sweeps"""
    config = OmegaConf.to_container(config)
    layout_name = config["ENV_KWARGS"]["layout"]

    def wrapped_train():
        wandb.init(project=config["PROJECT"])
        run_config = copy.deepcopy(config)
        for k, v in dict(wandb.config).items():
            run_config[k] = v
        run_config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]

        print("running experiment with params:", run_config)
        single_run(run_config)  # Use single_run directly

    sweep_config = {
        "name": "ppo_overcooked_sf",
        "method": "bayes",
        "metric": {"name": "returned_episode_returns", "goal": "maximize"},
        "parameters": {
            "NUM_ENVS": {"values": [32, 64, 128, 256]},
            "LR": {"values": [0.0005, 0.0001, 0.00005, 0.00001]},
            "FEATURE_LR": {"values": [0.0005, 0.0001, 0.00005]},
            "SF_LR": {"values": [0.0005, 0.0001, 0.00005]},
            "ACTIVATION": {"values": ["relu", "tanh"]},
            "UPDATE_EPOCHS": {"values": [2, 4, 8]},
            "NUM_MINIBATCHES": {"values": [2, 4, 8, 16]},
            "CLIP_EPS": {"values": [0.1, 0.2, 0.3]},
            "ENT_COEF": {"values": [0.0001, 0.001, 0.01]},
            "SF_COEF": {"values": [0.1, 0.5, 1.0]},
            "NUM_STEPS": {"values": [64, 128, 256]},
            "FEATURE_DIM": {"values": [32, 64, 128]},
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config,
        entity=config["ENTITY"],
        project=config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_train, count=1000)

def get_rollout(training_state: Union[TrainingState, Any], config: dict):
    """Generate rollout for visualization"""
    # Type checking
    if config["USE_SF"] and not isinstance(training_state, TrainingState):
        raise ValueError("Expected TrainingState when USE_SF is True")
        
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    network = SRActorCritic(
        action_dim=env.action_space().n,
        use_sf=config["USE_SF"],
        feature_dim=config["FEATURE_DIM"] if config["USE_SF"] else None,
        activation=config["ACTIVATION"]
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
            encoded_state = training_state.feature_state.apply_fn(
                training_state.feature_state.params,
                obs_batch
            )
            sf_features = training_state.sf_state.apply_fn(
                training_state.sf_state.params,
                encoded_state
            )
            pi, value = network.apply(
                training_state.policy_state.params,
                obs_batch,
                sf_features
            )
        else:
            pi, value = network.apply(
                training_state.policy_state.params,
                obs_batch
            )
        
        action = pi.sample(seed=key_act)
        env_act = unbatchify(action, env.agents, 1, env.num_agents)
        env_act = {k: v.squeeze() for k, v in env_act.items()}

        obs, state, reward, done, info = env.step(key_step, state, env_act)
        done = done["__all__"]
        state_seq.append(state)

    return state_seq

def unbatchify(batch, agents, batch_size, num_agents):
    """Utility function to unbatchify actions"""
    batch = batch.reshape(batch_size, num_agents, -1)
    return {agent: batch[:, i] for i, agent in enumerate(agents)}

@hydra.main(version_base=None, config_path="config", config_name="ippo_cnn_overcooked_sf")
def main(config):
    if config["TUNE"]:
        tune(config)
    else:
        single_run(config)

if __name__ == "__main__":
    main()
