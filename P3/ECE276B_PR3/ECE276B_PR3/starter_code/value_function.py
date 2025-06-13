import numpy as np


class ValueFunction:
    def __init__(self, T: int, ex_space, ey_space, etheta_space):
        self.T = T
        self.ex_space = ex_space
        self.ey_space = ey_space
        self.etheta_space = etheta_space
        self.value = np.zeros((T, len(ex_space), len(ey_space), len(etheta_space)))

    def copy_from(self, other):
        """
        Update the underlying value function storage with another value function
        """
        self.value = other.value.copy()

    def update(self, t, ex_idx, ey_idx, etheta_idx, target_value):
        """
        Update the value function at given states
        Args:
            t: time step
            ex_idx: x position error index
            ey_idx: y position error index
            etheta_idx: theta error index
            target_value: target value
        """
        t_idx = t % self.T
        self.value[t_idx, ex_idx, ey_idx, etheta_idx] = target_value

    def __call__(self, t, ex_idx, ey_idx, etheta_idx):
        """
        Get the value function results at given states
        Args:
            t: time step
            ex_idx: x position error index
            ey_idx: y position error index
            etheta_idx: theta error index
        Returns:
            value function results
        """
        t_idx = t % self.T
        return self.value[t_idx, ex_idx, ey_idx, etheta_idx]

    def copy(self):
        """
        Create a copy of the value function
        Returns:
            a copy of the value function
        """
        new_value = ValueFunction(self.T, self.ex_space, self.ey_space, self.etheta_space)
        new_value.value = self.value.copy()
        return new_value


class GridValueFunction(ValueFunction):
    """
    Grid-based value function
    """
    def __init__(self, T: int, ex_space, ey_space, etheta_space):
        super().__init__(T, ex_space, ey_space, etheta_space)


'''class FeatureValueFunction(ValueFunction):
    """
    Feature-based value function
    """
    # TODO: your implementation
    raise NotImplementedError'''


