import flax.linen as nn
import jax.numpy as jnp
import distrax
from typing import Optional

class SRActorCritic(nn.Module):
    """Actor-Critic that can use SR features"""
    action_dim: int
    use_sf: bool = False
    feature_dim: Optional[int] = None
    activation: str = "relu"
    
    @nn.compact
    def __call__(self, x, sf_features=None):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
            
        if self.use_sf and sf_features is not None:
            # Reshape sf_features to (1, 1, 1, 64) and broadcast to (1, 8, 5, 64)
            sf_features = sf_features.reshape((-1, 1, 1, sf_features.shape[-1]))
            sf_features = jnp.broadcast_to(sf_features, (x.shape[0], x.shape[1], x.shape[2], sf_features.shape[-1]))
            
            # Concatenate to get shape (1, 8, 5, 90) - combining original 26 features with 64 SF features
            x = jnp.concatenate([x, sf_features], axis=-1)
            
        # Actor head
        actor = nn.Dense(64)(x)
        actor = activation(actor)
        actor = nn.Dense(self.action_dim)(actor)
        pi = distrax.Categorical(logits=actor)
        
        # Critic head
        critic = nn.Dense(64)(x)
        critic = activation(critic)
        value = nn.Dense(1)(critic)
        
        return pi, value 