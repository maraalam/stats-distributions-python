import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .GenericDistribution import Distribution

class Binomial(Distribution):
    """ Implementation of the Binomial distribution. This distribution is used when there are exacly two mullually exclusive outcomes of a trial. There outcomes are appropiately labeled 'success' and 'failure'. The binomial distribution is used to obtain the probability of observing `x`successes in `N`trials, with the probability of success on a single trial dnoted by `p`. The binomial distribution assumes that `p` is fixed for all trials.

    
    It has methods for calculate mean, standard deviation and probability density function.

    Attributes
    ----------
    mean : float
        Representing the mean value of the distribution
    stdev : float
        Representing the standard deviation of the distribution
    data : list of floats
        Data set with mean `mean` and standard deviation `stdev`
    p : float
        representing the probability of an event occurring
    n : int
        number of trials

    Notes 
    -----

    """

    def __init__(self, n=10, prob=0.5, data=None, size=None):
        """
        Create a Binomial Distribution.

        Parameters
        ----------
        n : int
            number of trials
        
        p : float
            representing the probability of an event occurrin

        data : list of floats, optional
            Data set provided. Mean and stdev are calculated from it.
            
        size : int or tuple of ints, optional
            Data shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if mu and sigma are both scalars. Otherwise, np.broadcast(mu, sigma).size samples are drawn



        Raises
        ------
        ValueError  
            If `size` has a non-negative value.

        Notes
        -----
        This method will implicitily create a `data` set (list of floats) using `np.random.binomial` method if `data` is not provided.

        """
        
        

        if data is not None:
            self.data = data
            self.n = len(self.data)
            self.p = 1.0 * sum(self.data) / len(self.data)
            
            Distribution.__init__(self, self.calculate_mean(), self.calculate_stdev() )
            
        else:
            
            if size < 0: raise ValueError('size value must be non-negative')

            self.n = n
            self.p = prob

            self.sample(size)
            Distribution.__init__(self, self.__calculate_mean__(), self.__calculate_stdev__())

            


    def get_data(self):
        """Function to get the data.

        Parameters
        ----------
            None

        Returns
        -------
            data : list of floats
        """

        return self.data


    def getmean(self) -> float:
        """Function to get the mean of the data set.

        Parameters
        ----------
            None

        Returns
        -------
            float: mean of the data
        """

        return self.mean


    def getstdev(self) -> float:
        """Function to get the standard deviation of the data set.

        Parameters
        ----------
            None

        Returns
        -------
            float: stantard deviation of the data set

        """

        return self.stdev


    def pdf(self, k):
        """Returns the probability density function (PDF) of the distribution evaluated at a specified point (or list of points)`x`.

        Parameters
        ----------
            x (float): point for calculating the probability density function

        Returns
        -------
            float: probability density function output
        
        Notes
        -----

        """

        a = math.factorial(self.n) / (math.factorial(k) * (math.factorial(self.n - k)))
        b = (self.p ** k) * (1 - self.p) ** (self.n - k)
        
        return a * b


    def plot_histogram(self, title="Binomial Distribution Histogram", figsize=(7,6), xlabel="data", ylabel="count", rwidth=0.9, color='#BCC2B5',*args,**kwargs):
        """Function to output a histogram of the instance variable data using matplotlib pyplot library
        
        Parameters
        ----------
            title : string
                title of the histogram
            figsize : tuple
                size of the plot
            xlabel : string
                label of the x axis
            ylabel : string
                label of the y axis
            rwidth : float
                width of each row (default 0.9)
            color : string
                color of the bars


        Returns
        -------
            None

        """

        
        plt.subplots(figsize=figsize, dpi=100)
        plt.title(title)
        sns.histplot(self.data, kde=False, color=color,*args,**kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()


    def sample(self,size=None):
        """
        TODO
        """
        self.data = np.random.binomial(n=self.n, p=self.p, size=size)
        return self.data


    def __calculate_mean__(self) -> float:
        """Function to calculate the mean of the data set.

        Parameters
        ----------
            None

        Returns
        -------
            float: mean of the data set
        
        """

        mean = self.p * self.n

        return mean


    def __calculate_stdev__(self) -> float:
        """Function to calculate the standard deviation of the data set.

        Parameters
        ----------
            None

        Returns
        -------
            float: stdev of the data set
        
        """

        stdev = math.sqrt(self.n * self.p * (1 - self.p))

        return stdev


    def __add__(self, other):

        """Function to add together two Binomial Distributions
        
        Parameters
        ----------
            other (Binomial): Binomial instance

        Returns
        -------
            Binomial: Binomial distribution
        """

        try:
            assert self.p == other.p, 'p values are not equal'
        except AssertionError as error:
            raise
        
        result = Binomial(self.n + other.n, self.p) 
        return result


    def __repr__(self) -> str:
        """Function to output the characteristics of the Binomial instance
        
        Parameters
        ----------
            None

        Returns
        -------
            string: characteristics of the Binomial

        """
        return f"mean {self.mean}, standard deviation {self.stdev}, p {self.p}, n {self.n}"



