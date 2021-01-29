# -*- coding: utf-8 -*-
"""
An ODL callback that is applied at specified iteration numbers.
"""
from inspect import signature
from odl.solvers.util.callback import Callback

class CallbackApplyAfter(Callback):
    def __init__(self, function, call_after_iters=None):
        """Initialize a new instance.

        Parameters
        ----------
        function : callable
            Callable to apply after the given iterations.
            Allowed signatures:
                ``function(result)``
                ``function(result, iters)`` (where `iters` is of type int)
        call_after_iters : list of int, optional
            Numbers of iterations after which `function` should be called.
        """
        self.function = function
        parameters = signature(self.function).parameters
        if len(parameters) == 2:
            iters_param_name = list(parameters.items())[1][0]
            def call_function(result):
                kwargs = {iters_param_name: self.iter + 1}
                self.function(result, **kwargs)
        else:
            call_function = self.function
        self.call_function = call_function
        self.call_after_iters = (call_after_iters
                                  if call_after_iters is not None else [])
        self.iter = 0

    def __call__(self, result):
        """Call :attr:`function`."""
        if (self.iter + 1) in self.call_after_iters:
            self.call_function(result)
        self.iter += 1

    def reset(self):
        """Reset iteration number."""
        self.iter = 0
