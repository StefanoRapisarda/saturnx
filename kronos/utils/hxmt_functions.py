import numpy as np

def pi2en(pi,instrument='LE'):

    if instrument == 'LE':
        a = 13./1536 
        b = 0.1
    elif instrument == 'ME':
        a = 60./1024
        b = 3
    elif instrument == 'HE':
        a = 370./256
        b = 15

    return a*pi+b