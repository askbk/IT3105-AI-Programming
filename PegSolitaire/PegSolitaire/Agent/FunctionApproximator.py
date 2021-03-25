class FunctionApproximator:
    """
    Function approximator abstract class
    """

    def __init__(self, eligibility_decay_rate, discount_factor, learning_rate):
        pass

    def get_value(self, state):
        raise NotImplementedError()

    def update(self, states, td_error):
        raise NotImplementedError()

    def reset_eligibilities(self):
        raise NotImplementedError()