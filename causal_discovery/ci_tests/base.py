from abc import ABC, abstractmethod

class CITest(ABC):

    @abstractmethod
    def is_independent(self, X, i, j, S, alpha):
        pass
