import jax
import jax.numpy as jnp

# Create a simple array operation on GPU
x = jnp.ones((1000, 1000))
print("JAX devices:", jax.devices())
print(x @ x.T)
