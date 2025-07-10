"""
This python file is meant to be a placeholder for users who do not have cupy installed
This way engine files can have @jit.rawkernel decorators freely for GPU-related functions, but will not affect the loading of these engines
    for CPU-only users
"""

class JitPlaceholder:
    def rawkernel(self, *args, **kwargs):
        def inner(func):
            return func
        return inner
    
    def grid(self, *args, **kwargs):
        return 0

jit = JitPlaceholder()