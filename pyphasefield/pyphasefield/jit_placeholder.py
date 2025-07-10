"""
This python file is meant to be a placeholder for users who do not have cupy installed
This way engine files can have @cuda.jit decorators freely for GPU-related functions, but will not affect the loading of these engines
    for CPU-only users
"""

def rawkernel(*args, **kwargs):
    def inner(func):
        return func
    return inner