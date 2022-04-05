import time
import functools

def timer(func):

    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        start_time = time.perf_counter()
        value = func(*args,**kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper

def debug(func):

    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k} = {v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")
        value = func(*args,**kwargs)
        try:
            print(f"{func.__name__!r} returned ({type(value)})\n{value!r}")
        except:
             print(f"{func.__name__!r} returned a complex value ({type(value)})")
        return value

    return wrapper
