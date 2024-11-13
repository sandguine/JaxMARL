import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from typing import Optional, Callable
import numpy as np
import optax

class FeatureEncoder(nn.Module):
    """Generic Feature Encoder that can work with different input types"""
    num_features: int
    encoder_type: str = "cnn"  # or "mlp" for different architectures
    activation: str = "relu"
    
    def setup(self):
        if self.activation == "relu":
            self.activation_fn = nn.relu
        else:
            self.activation_fn = nn.tanh
            
    @nn.compact
    def __call__(self, x):
        if self.encoder_type == "cnn":
            x = self._cnn_encoder(x)
        else:
            x = self._mlp_encoder(x)
        return x
    
    def _cnn_encoder(self, x):
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation_fn(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation_fn(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.num_features)(x)
        return x
    
    def _mlp_encoder(self, x):
        x = nn.Dense(128)(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.num_features)(x)
        return x

class SuccessorFeatureNetwork(nn.Module):
    """Generic Successor Feature Network"""
    num_features: int
    activation: str = "relu"
    hidden_dims: tuple = (128, 128)
    
    def setup(self):
        if self.activation == "relu":
            self.activation_fn = nn.relu
        else:
            self.activation_fn = nn.tanh
    
    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(
                dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0)
            )(x)
            x = self.activation_fn(x)
        
        x = nn.Dense(
            self.num_features,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        return x

class SRModule:
    """Wrapper class for SR components with training logic"""
    def __init__(self, feature_dim, encoder_type="cnn", activation="relu"):
        self.feature_encoder = FeatureEncoder(feature_dim, encoder_type, activation)
        self.sf_network = SuccessorFeatureNetwork(feature_dim, activation)
    
    def compute_sf_loss(self, encoded_state, next_encoded_state, done, gamma):
        """Compute SR prediction loss"""
        sf_target = encoded_state + gamma * (1 - done) * next_encoded_state
        return optax.huber_loss(self.sf_network(encoded_state), sf_target).mean() 