
class IR:
    """
    This class encapsulates a stencil's intermediate representation.
    """
    def __init__(self, stencil_configs, graph):
        self.stencil_configs = stencil_configs
        self.graph = graph
        self.assignment_statements_graphs = None
        self.computation_func = None
        self.minimum_halo = None

    def __getattribute__(self, name):
        """
        This method checks that the attribute being accessed is valid, i.e. not None. In fact, if the sequence
        of compilation passes being executed is malformed, the attribute being accessed might be invalid because
        the pass that initializes it has not been executed yet. In such a case this method fires an assertion.
        """
        value = super().__getattribute__(name)
        assert value is not None,  ("Attempted to access an invalid attribute of the intermediate representation. " +
                                    "Hint: is the sequence of compilation passes being executed correct?")
        return value
