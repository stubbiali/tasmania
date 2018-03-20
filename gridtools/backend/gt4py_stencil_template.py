import numpy as np
from gridtools.stencil import Stencil, MultiStageStencil


class {{ stencil_name }}AutoStencil (MultiStageStencil):
    def __init__(self, temps_shape=None):
        super().__init__()

        {% if temps -%}
        #
        # temporary data fields to share data among the different stages
        #
        {% for t in temps -%}
        self.{{ t }} = np.zeros(temps_shape)
        {% endfor -%}
        {%- endif -%}

    {% for stg in stages %}
    def stage_{{ stg["name"] }}(self,
            {%- for arg in stg["args"] -%}
            {{ arg }}
                {%- if not loop.last -%}
                , {% endif -%}
            {% endfor -%}):
        for p in self.get_interior_points({{stg["outputs"][0]}}):
            {% for exp in stg["expressions"] -%}
            {{ exp }}
            {% endfor -%}
    {%- endfor %}

    @Stencil.kernel
    def kernel(self,
            {%- for i in stencil_args -%}
            {{ i }}
                {%- if not loop.last -%}
                , {% endif -%}
            {%- endfor -%}):

        {%- for stg in stages %}
        self.stage_{{ stg["name"] }}(
                {%- for arg in stg["args"] -%}
                {{ arg }}={% if arg in temps -%}
                            self.
                            {%- endif -%}
                            {{ arg }}
                    {%- if not loop.last -%}
                    , {% endif -%}
                {% endfor -%})
        {% endfor -%}