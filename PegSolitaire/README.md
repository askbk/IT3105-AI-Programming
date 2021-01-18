# Peg Solitaire with Reinforcement Learning

This project implements Temporal Difference learning with Eligibility Traces using the Actor-Critic model in order to play Peg Solitaire.
The model implements two different versions of the Critic: One is a simple table lookup, the other uses a neural network as a function approximator.

## Testing

Unit tests may be run with `pytest`.
To produce a more detailed coverage report, `pytest --cov --cov-report: term`
For more details on coverage reporting, please see the [pytest-cov documentation](https://pytest-cov.readthedocs.io/en/latest/reporting.html).

## Architecture

The `Board` class and the `Position` class implement all necessary game logic.

## ACMAgent

### Public interface

**ACMAgent**(actor_discount_factor=0.9, actor_eligibility_decay_rate=0.9,actor_learning_rate=0.01, critic_function="table", critic_nn_dimensions=None,critic_learning_rate=0.9, critic_eligibility_decay_rate=0.9, critic_discount_factor=0.9, initial_epsilon=0.05, epsilon_decay_rate=0.9)
Returns a new Agent instance.

**choose_action**(state, possible_actions, reward)
Gives reward for current state and returns next action to take.

## Actor

### Public interface

**Actor**(actor_discount_factor=0.9,actor_eligibility_decay_rate=0.9,actor_learning_rate=0.01,initial_epsilon=0.05,epsilon_decay_rate=0.9)
Constructor.

**choose_action**(state, possible_actions)
Returns an action from possible_actions.

**update_policy**(state, action, temporal_difference_error)
Returns a new Actor instance with an updated policy function.

**update_eligibility**(state, action)
Returns a new Actor instance with updated eligibility for the state-action pair.

## Critic

### Public interface

**Critic**(critic_function="table", critic_nn_dimensions=None,critic_learning_rate=0.9, critic_eligibility_decay_rate=0.9,critic_discount_factor=0.9, value_function=None, eligibilities=None)
Constructor.

**get_temporal_difference_error**(old_state, new_state, reward)
Calculates the temporal difference error.

**update**(old_state, states, temporal_difference_error)
Returns a new Critic instance with an updated value function.
