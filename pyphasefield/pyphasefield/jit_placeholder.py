"""
This python file is meant to be a placeholder for users who do not have cuda installed
This way engine files can have @cuda.jit decorators freely for GPU-related functions, but will not affect the loading of these engines
    for CPU-only users
"""

def jit(*args, **kwargs):
    def inner(func):
        return func
    return inner