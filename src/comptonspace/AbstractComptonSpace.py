from abc import ABC, abstractmethod

class AbstractComptonSpace(ABC):
    """
    This class is an abstract class for all compton data space class
    """

    @abstractmethod
    def getTotalBinNums(self):
        """
        Abstractmethod - intend to return the total number of bins on that compton data space
        ------------------------------------------------------------------------------------
        :return: int
        """
        pass

    @abstractmethod
    def sampleSinglePointOnXY(self):
        """
        Abstractmethod - intend to sample a random point on the XY plane regardless of Z
        ------------------------------------------------------------------------------------
        :return: (X, Y)
        """
        pass