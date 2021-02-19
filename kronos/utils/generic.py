import numpy as np
from datetime import datetime

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

