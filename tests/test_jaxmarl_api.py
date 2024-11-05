""" 
Test auto reseting works as expected
"""
import jax 
import jax.numpy as jnp
from jaxmarl import make

def test_auto_reset_to_specific_state():
    
    def _test_leaf(x, y, outcome=True):
        x = jnp.array_equal(x, y)
        assert x==outcome
    
    env = make("MPE_simple_spread_v3")
    print("Environment created:", env)
    
    rng = jax.random.PRNGKey(0)
    rng, rng_reset1, rng_reset2 = jax.random.split(rng, 3)
    print("Random keys generated:", rng_reset1, rng_reset2)
    
    _, state1 = env.reset(rng_reset1)
    _, state2 = env.reset(rng_reset2)
    print("Initial states after reset:", state1, state2)

    # normal step
    rng, rng_act = jax.random.split(rng)
    rng_act = jax.random.split(rng_act, env.num_agents)
    actions = {a: env.action_space(a).sample(rng_act[i]) for i, a in enumerate(env.agents)}
    _, next_state, _, dones, _ = env.step(rng, state1, actions, reset_state=state2)
    print("Actions:", actions)
    print("Next state after step:", next_state)
    print("Dones:", dones)
    assert not dones["__all__"]
    assert not jnp.array_equal(state2.p_pos, next_state.p_pos)
    
    # auto reset to specific state
    state1 = state1.replace(
        step = env.max_steps,
    )
    print("State1 at max steps:", state1)

    rng, rng_act = jax.random.split(rng)
    rng_act = jax.random.split(rng_act, env.num_agents)
    actions = {a: env.action_space(a).sample(rng_act[i]) for i, a in enumerate(env.agents)}
    _, next_state, _, dones, _ = env.step(rng, state1, actions, reset_state=state2)
    print("Next state after auto-reset:", next_state)
    print("Dones after auto-reset:", dones)
    assert dones["__all__"]
    jax.tree_util.tree_map(_test_leaf, state2, next_state)

print("JAX devices:", jax.devices())
print("Ran test_jaxmarl_api.py")

# call the function to test that it's running
test_auto_reset_to_specific_state()