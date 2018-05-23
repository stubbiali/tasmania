
class NGCompiler:
    def __init__(self, passes):
        self._passes = passes
        self._minimum_halo = None

    def compile(self, stencil_configs):
        """
        Compiles a computation starting from the stencil's configuration parameters specified by the user.
        The result of the compilation is a function that implements the computation.

        Parameters
        ----------
        stencil_configs:
            The stencil's configuration parameters specified by the user.

        Returns
        -------
            A function that implements the computation defined by the stencil's configuration parameters.
        """
        ir = stencil_configs
        for p in self._passes:
            ir = p.process(ir)
        self._minimum_halo = ir.minimum_halo
        return ir.computation_func

    def get_extent(self):
        return self._minimum_halo

