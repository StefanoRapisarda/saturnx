import numpy as np
from datetime import datetime

import re
import pathlib
import math

import matplotlib
import matplotlib.pyplot as plt

def round_half_up(n, decimals=0):
    '''
    From https://realpython.com/python-rounding/
    '''
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

def str_title(title,total_characters=72,symbol='*'):
    '''
    Returns a string of length total_characters with title in the 
    middle and symbol filling the rest of space.
    '''

    len_title = len(title)
    if len_title%2 != 0:
        len_title += 1

    num_of_symbols = int((total_characters - len_title - 2)/2)
    result = '{:^{l}} {:^{c}} {:^{r}}'.\
        format(symbol*num_of_symbols,title,symbol*num_of_symbols,
            l = num_of_symbols,c = len_title, r = num_of_symbols)

    return result


def print_meta_data(cls):
    for key, value in cls.meta_data.items():
        if type(value) == dict:
            print('{}:'.format(key))
            for key2,value2 in value.items():
                print('--- {}: {}'.format(key2,value2))
        else:
            print('{}: {}'.format(key,value))

def plt_color(cmap_name='tab10',hex=False):
    nc = plt.get_cmap(cmap_name).N
    ccolors = plt.get_cmap(cmap_name)(np.arange(nc, dtype=int))
    if hex:
        ccolors = [matplotlib.colors.rgb2hex(c) for c in ccolors]
    return ccolors

def my_cdate():
    now = datetime.utcnow()
    date = f'{now.year}-{now.month}-{now.day},{now.hour}:{now.minute}:{now.second}'
    return date

def clean_expr(expr,characters='=+-/*()[]}{><!&|^~',other_words=['not','and','xor','or']):
    '''

    NOTES
    -----
    2021 02 18, Stefano Rapisarda (Uppsala)
        It is importand that xor is listed before or
    '''
    new_expr = expr
    for char in characters:
        new_expr = new_expr.replace(char,' ')
    for word in other_words:
        new_expr = new_expr.replace(word,' ')
    return new_expr

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def chunks(in_obj,n_chunks):  
    '''
    This small function was designed for plotting purposes
    It splits a new list or an existing one in n_chunks

    PARAMETERS
    ----------
    in_obj: integer or list
        If in_obj is an interger, a list of integer between 0 and 
        in_obj-1 will be splitted in n_chunks.
        If in_obj is a list, the list will be splitted in n_chunks.
    n_chunks: integer
        Number of chunks

    HISTORY
    -------
    Stefano Rapisarda, 2019 06 21 (Shanghai), creation date
    '''

    if type(in_obj) == type([]) or type(in_obj) == type(np.array([])):
        n = len(in_obj)
        array = in_obj
    elif type(in_obj) == type(1):
        n = in_obj
        array = list(range(in_obj))
    else:
        print('Wrong object type')
        return

    if n%n_chunks == 0:
        sub = [[array[i+n_chunks*j] for i in range(n_chunks)] for j in range(int(n/n_chunks))]
    else:
        sub = [[array[i+n_chunks*j] for i in range(n_chunks)] for j in range(int(n/n_chunks))]+\
              [[array[n+i-n%n_chunks] for i in range(n%n_chunks)]]

    return sub

