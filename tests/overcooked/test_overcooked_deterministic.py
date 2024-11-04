import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked import Overcooked

# Initialize the environment
env = Overcooked()

def test_deterministic_rollout():
    # Set a random seed for reproducibility
    rng = jax.random.PRNGKey(0)
    rng, rng_reset = jax.random.split(rng)
    
    # Reset the environment and get the initial state
    _, state = env.reset(rng_reset)
    assert state is not None, "Initial state should not be None"

    # Define a series of fixed actions for each agent
    # Assuming there are two agents, you can replace with appropriate actions for each timestep
    deterministic_actions = [
        {0: 0, 1: 1},  # Actions for agents at step 1
        {0: 2, 1: 3},  # Actions for agents at step 2
        # Add more fixed actions as needed
    ]
    
    # Perform steps using the fixed actions
    for step, actions in enumerate(deterministic_actions):
        print(f"\n--- Step {step + 1} ---")
        
        # Step in the environment with deterministic actions
        rng, state, reward, done, _ = env.step(rng, state, actions)
        
        # Check the state, reward, and done flag are as expected
        # Replace expected_state, expected_reward, and expected_done with actual expected values for each step
        expected_state = None  # Replace with expected state for this step
        expected_reward = {0: 1.0, 1: 1.0}  # Example reward for each agent
        expected_done = {"__all__": False}  # Example done flag
        
        assert state == expected_state, f"Unexpected state at step {step + 1}"
        assert reward == expected_reward, f"Unexpected reward at step {step + 1}"
        assert done == expected_done, f"Unexpected done flag at step {step + 1}"

        # Terminate test if environment ends
        if done["__all__"]:
            print("Environment terminated for all agents.")
            break

    print("Deterministic rollout test completed successfully.")
