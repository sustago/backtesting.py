import numpy as np

from skopt.sampler import Lhs
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, Configuration, \
    Constant
from openbox.utils.util_funcs import check_random_state


class Sampler(object):
    """
    Generate samples within the specified domain (which defaults to the whole config space).

    Users should call generate() which auto-scales the samples to the domain.

    To implement new design methodologies, subclasses should implement _generate().
    """

    def __init__(self, config_space: ConfigurationSpace, size, random_state=None):
        """
        Parameters
        ----------
        config_space : ConfigurationSpace
            ConfigurationSpace to do sampling.

        size : int N
            Number of samples.
        """
        self.config_space = config_space

        self.search_dims = []
        for i, param in enumerate(config_space.get_hyperparameters()):
            if isinstance(param, Constant):
                self.search_dims.append([1.0])
            elif isinstance(param, UniformFloatHyperparameter):
                self.search_dims.append((0.0, 1.0))
            elif isinstance(param, UniformIntegerHyperparameter):
                self.search_dims.append((0.0, 1.0))
            else:
                raise NotImplementedError('Only Integer and Float are supported in %s.' % self.__class__.__name__)

        self.size = size
        self.rng = check_random_state(random_state)

    def set_params(self, **params):
        """
        Set the parameters of this sampler.

        Parameters
        ----------
        **params : dict
            Generator parameters.
        Returns
        -------
        self : object
            Generator instance.
        """
        if not params:
            return self
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def generate(self, return_config=True):
        """
        Create samples in the domain specified during construction.

        Returns
        -------
        configs : list
            List of N sampled configurations within domain. (return_config is True)

        X : array, shape (N, D)
            Design matrix X in the specified domain. (return_config is False)
        """
        X = self._generate()
        X = np.asarray(X)

        if return_config:
            configs = [Configuration(self.config_space, vector=x) for x in X]
            return configs
        else:
            return X

    def _generate(self):
        """
        Create unscaled samples.

        Returns
        -------
        X : array, shape (N, D)
            Design matrix X in the config space's domain.
        """
        raise NotImplementedError()


class LatinHypercubeSampler(Sampler):
    """
    Latin hypercube sampler.
    """

    def __init__(self, config_space: ConfigurationSpace,
                 size, criterion='maximin', iterations=10000,
                 random_state=None):
        """
        Parameters
        ----------
        config_space : ConfigurationSpace
            ConfigurationSpace to do sampling.

        size : int N
            Number of samples.

        criterion : str or None, default='maximin'
            When set to None, the latin hypercube is not optimized

            - 'correlation' : optimized latin hypercube by minimizing the correlation
            - 'maximin' : optimized latin hypercube by maximizing the minimal pdist
            - 'ratio' : optimized latin hypercube by minimizing the ratio
              `max(pdist) / min(pdist)`

        iterations : int
            Define the number of iterations for optimizing latin hypercube.
        """
        super().__init__(config_space, size, random_state)
        self.criterion = criterion
        self.iterations = iterations

    def _generate(self):
        lhs = Lhs(criterion=self.criterion, iterations=self.iterations)
        X = lhs.generate(self.search_dims, self.size, random_state=self.rng)
        X = np.asarray(X)  # returns a list in scikit-optimize==0.9.0
        return X
