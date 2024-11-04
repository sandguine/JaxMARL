import jax
import pytest
import jax.numpy as jnp
from jaxmarl.environments.overcooked import Overcooked 

# Initialize the environment
env = Overcooked()
print("Environment created:", env)

@pytest.mark.parametrize("steps", [2])  # Define a few steps for the deterministic test
def test_deterministic_rollout(steps):

    # Set random seed for reproducibility
    rng = jax.random.PRNGKey(0)
    rng, rng_reset = jax.random.split(rng)
    print("Random keys generated:", rng, rng_reset)
    
    # Reset the environment and verify initial state
    _, state = env.reset(rng_reset)
    assert state is not None, "Failed to reset environment: state is None"
    print("Initial state:", state)

    # Define deterministic actions for each step
    deterministic_actions = [
        {"agent_0": 0, "agent_1": 1},  # Actions for step 1
        {"agent_0": 2, "agent_1": 3},  # Actions for step 2
    ]
    
    for step, actions in enumerate(deterministic_actions):
        print(f"\n--- Step {step + 1} ---")
        
        # Step in the environment with deterministic actions
        rng, state, reward, dones, _ = env.step(rng, state, actions)
        
        # State Assertions: Check that the state is not None and key attributes are valid
        assert state is not None, "State is None after step"
        assert isinstance(state.agent_pos, jnp.ndarray), "Agent positions are not a jnp.ndarray"
        
        # Convert JAX array rewards to Python floats for comparison
        reward_as_floats = {agent: reward[agent].item() for agent in reward}

        # Example expected reward and done values (modify based on expected results)
        expected_reward = {"agent_0": 0.0, "agent_1": 0.0}  # Modify these values as needed
        
        # Convert dones to standard Python bools for comparison
        dones_as_bools = {agent: dones[agent].item() for agent in dones}
        expected_done = {"agent_0": False, "agent_1": False, "__all__": False}  # Modify as needed

        # Assert rewards and done flags match expectations
        assert reward_as_floats == expected_reward, f"Unexpected reward at step {step + 1}"
        assert dones_as_bools == expected_done, f"Unexpected dones at step {step + 1}"

        # Check if the environment has terminated for all agents
        if dones["__all__"].item():
            print("Environment terminated for all agents.")
            break

    print("Deterministic rollout test completed successfully.")