import numpy as np
from datetime import datetime

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

