import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

class ActionPredictor(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)
        return nn.softmax(x)

def create_train_state(rng, num_actions):
    model = ActionPredictor(num_actions)
    params = model.init(rng, jnp.ones([1, 10]))  # Assuming input features are of size 10
    tx = optax.adam(learning_rate=0.001)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def predict_action(state, model, params):
    logits = model.apply(params, state)
    return jnp.argmax(logits, axis=-1)