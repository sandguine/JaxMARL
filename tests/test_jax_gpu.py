import jax
import jax.numpy as jnp

# Create a simple array operation on GPU
x = jnp.ones((1000, 1000))
print("JAX devices:", jax.devices())
print("Tensor created in ", x.addressable_data(0).devices())
print("Tensor created in ", x.devices())
print(x @ x.T)
print(x @ x)

assert jnp.allclose(x @ x, x @ x.T) 