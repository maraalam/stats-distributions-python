class Distribution:

    def __init__(self, data = [], mu=0, sigma=1):
        """ Generic distribution class for calculating and visualizing a probability distribution.

        Attributes
        ----------
        mean : float
            Representing the mean value of the distribution
        stdev : float
            Representing the standard deviation of the distribution
        data: list of floats
            Data set with mean `mean` and standard deviation `stdev`

        """
        self.data = data
        self.mean = mu
        self.stdev = sigma
        
