# Peg Solitaire with Reinforcement Learning

This project implements Temporal Difference learning with Eligibility Traces using the Actor-Critic model in order to play Peg Solitaire.
The model implements two different versions of the Critic: One is a simple table lookup, the other uses a neural network as a function approximator.

## Testing

Unit tests may be run with `pytest`.
To produce a more detailed coverage report, `pytest --cov --cov-report: term`
For more details on coverage reporting, please see the [pytest-cov documentation](https://pytest-cov.readthedocs.io/en/latest/reporting.html).

## Architecture

The `Board` class and the `Position` class implement all necessary game logic.
