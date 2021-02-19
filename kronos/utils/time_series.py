import numpy as np

def poi_events(tres=1, nbins=1000, cr=1):
    '''
    Creates a numpy array of Poisson distributed events

    PARAMETERS
    ----------
    tres: float
        Time resolution
    
    nbins: int
        Number of bins with length equal to the time resolution

    cr: float
        Count rate [counts/seconds]

    RETURNS
    -------
    time: np.array
        Array of time tags

    HISTORY
    -------
    2021 02 18, Stefano Rapisarda, Uppsala (creation date)
        I made this for testing purposes

    NOTES
    -----
    The number of bins refers to the number of bins you would obtain
    if grouping the time arrival of photons with np.histogram and the
    provided time resolution
    '''

    # This is the histogram
    poi = np.array([np.random.poisson(tres*cr) for i in range(nbins)])
    time = np.concatenate([np.random.uniform(0+i*tres,(i+1)*tres,poi[i]) for i in range(nbins)],axis=0)
    time = sorted(time)

    return time