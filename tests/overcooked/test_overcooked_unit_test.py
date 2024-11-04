"""
Unit test for the Overcooked environment's ability to reset and step with random actions.
"""

import jax
import pytest
import jax.numpy as jnp
from jaxmarl.environments.overcooked import Overcooked 

# Initialize the environment
env = Overcooked()
print("Environment created:", env)

@pytest.mark.parametrize("steps", [10])
def test_random_rollout(steps):

    # Set random seed for reproducibility
    rng = jax.random.PRNGKey(0) 
    rng, rng_reset = jax.random.split(rng)
    print("Random keys generated:", rng, rng_reset)
    
    # Reset the environment and verify initial state is returned
    _, state = env.reset(rng_reset)
    assert state is not None, "Failed to reset environment: state is None"
    print("Initial state:", state)

    # Perform a series of steps with random actions
    for step in range(steps):
        print(f"\n--- Step {step + 1} ---")
        rng, rng_act = jax.random.split(rng)
        rng_act = jax.random.split(rng_act, env.num_agents)
        
        # Generate random actions
        actions = {a: env.action_space(a).sample(rng_act[i]) for i, a in enumerate(env.agents)}
        assert actions, "Failed to generate actions: actions is empty or None"
        print("Actions generated:", actions)
        
        # Step in the environment
        _, state, reward, dones, _ = env.step(rng, state, actions)
        print("State after step:", state)
        print("Reward after step:", reward)
        print("Dones after step:", dones)
        
        # Assertions to verify expected behavior after each step
        assert state is not None, "State is None after step"
        assert isinstance(reward, dict), "Reward is not a dictionary"
        assert isinstance(dones, dict), "Dones is not a dictionary"

        # Ensure all done values are boolean-compatible
        assert all(isinstance(done, (bool, jnp.ndarray)) and done.dtype == jnp.bool_ for done in dones.values()), f"Dones contains non-boolean values: {dones}"


        # Check if the environment has terminated for all agents
        if dones["__all__"]:
            print("Environment terminated for all agents.")
            break

    print("Random rollout test completed successfully.")