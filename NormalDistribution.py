import math
import matplotlib.pyplot as plt
import numpy as np
from .GenericDistribution import Distribution

class Normal(Distribution):
    """ Implementation of the Normal (Gaussian) distribution. It can be used to create a normal distribution from your data or from a given `\mean`and `\stdev`. 

    It has methods for calculate mean, standard deviation, probability density function and cummulate density funcition.

    Attributes
    ----------
    mean : float
        Representing the mean value of the distribution
    stdev : float
        Representing the standard deviation of the distribution
    data: list of floats
        Data set with mean `mean` and standard deviation `stdev`

    Notes
    -----
    A Normal Distribution or Gaussian Distribution is a type of continuous probability distribution for a real-valued random variable. Is by far one of the most important probability distribution. One of the main reasons for that is the Central Limit Theorem (CLT).

    The CLT states that if you add a large number of random variables, the distribution of the sum will be approbimately normal under certain conditions. Its importance comes from the fact that meny random variables in real life can be expressed in that way. 
    
    """

    def __init__(self, mu=0, sigma=1, size=None, data=None):
        """
        Create a Normal (Gaussian) Distribution using the given mean (:math:`\mu`) and stdev (:math:`\sigma`) values.

        Parameters
        ----------
        mu : float
            Mean ("centre") value of the distribution
        sigma : float
            Standard deviation (spread or "width") for this distribution. Must be non-negative

        data : list of floats, optional
            Data set provided. Mean and stdev are calculated from it.
            
        size : int or tuple of ints, optional
            Data shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if mu and sigma are both scalars. Otherwise, np.broadcast(mu, sigma).size samples are drawn



        Raises
        ------
        ValueError  
            If `sigma` has a non-negative value.

        Notes
        -----
        This method will implicitily create a `data` set (list of floats) using `np.random.normal` method if `data` is not provided.

        """
        
        if data is not None:
            self.mean = self.__calculate_mean__(data)
            self.stdev = self.__calculate_sigma__(data)
            self.data = data
            
        else:
            if sigma<0: raise ValueError('sigma value must be non-negative')

            self.mean = mu
            self.stdev = sigma

            self.sample(size)


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


    def mean(self) -> float:
        """Function to get the mean (`sigma`) of the data set.

        Parameters
        ----------
            None

        Returns
        -------
            float: mean of the data
        """

        return self.mean


    def stdev(self) -> float:
        """Function to get the standard deviation of the data set.

        Parameters
        ----------
            None

        Returns
        -------
            float: stantard deviation of the data set

        """

        return self.stdev


    def pdf(self, x):
        """Returns the probability density function (PDF) of the distribution evaluated at a specified point (or list of points)`x`.

        Parameters
        ----------
            x (float): point for calculating the probability density function

        Returns
        -------
            float: probability density function output
        
        Notes
        -----
        The probability density function (pdf) for normal distribution is

        .. math::

            f(x;\mu,\sigma^2) = \frac{\exp(-x^2/2)}{\sqrt{2\pi}}
            
        where,
        * :math:`\mu` is the mean or expectation of the distribution (where the function is centred) 
        * :math:`\sigma^2` is the variance
        * :math:`x` is a real number

        """

        return (1.0/ (20 * math.sqrt(2*math.pi))) * np.exp(-0.5 * ((x - 15) / 20) ** 2)


    def cdf(self, x):
        """Returns the cummulative density function (CDF) of the distribution evaluated at a specified point `x`.

        Parameters
        ----------
            x (float): point for calculating the probability density function

        Returns
        -------
            float: probability cummulative function output
        
        Notes
        -----
        The cummulative density function (cdf) for normal distribution is:

        .. math::

            f(x;\mu,\sigma) = 1/2*[1+erf(\frac{x-\mu}{\sigma*\sqrt{2\pi}})]
        where,
        * :math:`\mu` is the mean or expectation of the distribution (where the function is centred) 
        * :math:`\sigma` is the standard deviation
        * :math:`x` is a real number
        
        """
        return 0.5 * (1 + math.erf((x - self.mean) / (self.stdev * math.sqrt(2))))


    def plot_histogram(self, title="Normal Distribution Histogram", figsize=(7,6), xlabel="data", ylabel="count", rwidth=0.9, color='#BCC2B5',*args,**kwargs):
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
        plt.hist(self.data,rwidth=rwidth, color=color,*args,**kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()


    def sample(self,size=None):
        """
        TODO
        """
        self.data = np.random.normal(self.mean,self.stdev,size)
        return self.data


    def __calculate_mean__(self,data) -> float:
        """Function to calculate the mean of the data set.

        Parameters
        ----------
            data : list of floats
                Data set provided. Mean and stdev are calculated from it.

        Returns
        -------
            float: mean of the data set
        
        """

        mu = 1.0 * sum(data) / len(data)

        return mu


    def __calculate_sigma__(self,data) -> float:
        """Function to calculate the standard deviation of the data set.

        Parameters
        ----------
            data : list of floats
                Data set provided. Mean and stdev are calculated from it.

        Returns
        -------
            float: stdev of the data set
        
        """

        sigma = 0

        for d in data:
            sigma += (d-self.mean) ** 2

        sigma = math.sqrt(sigma / len(data))

        return sigma


    def __add__(self, other):

        """Function to add together two Normal Distributions
        
        Parameters
        ----------
            other (Normal): Normal instance

        Returns
        -------
            Normal: Normal distribution
        """

        result = Normal(self.data +other.data)
        result.mean = self.mean + other.mean
        result.stdev = math.sqrt(self.stdev ** 2 + other.stdev ** 2)

        return result


    def __repr__(self) -> str:
        """Function to output the characteristics of the Normal instance
        
        Parameters
        ----------
            None

        Returns
        -------
            string: characteristics of the Normal

        TODO
        """
        return "mean {}, standard deviation {}".format(self.mean, self.stdev)