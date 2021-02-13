class EligibilityTable:
    def __init__(self, discount_factor, eligibility_decay_rate, _eligibilities=None):
        self._discount_factor = discount_factor
        self._eligibility_decay_rate = eligibility_decay_rate
        self._eligibilities = dict() if _eligibilities is None else _eligibilities

    @staticmethod
    def _new(old, eligibilities):
        return EligibilityTable(
            old._discount_factor,
            old._eligibility_decay_rate,
            _eligibilities=eligibilities,
        )

    def get_eligibility(self, state, was_previous):
        if was_previous:
            return 1

        try:
            return self._eligibilities[str(state)]
        except KeyError:
            return 0

    def update_eligibilities(self, states):
        return EligibilityTable._new(
            self,
            self._eligibilities
            | {
                str(state): self._discount_factor
                * self._eligibility_decay_rate
                * self.get_eligibility(state, was_previous=(state == states[-1]))
                for state in states
            },
        )

    def reset(self):
        return EligibilityTable._new(self, dict.fromkeys(self._eligibilities, 0))