import traitlets
import numpy as np
try:
    import ipdb
except traitlets.config.configurable.MultipleInstanceError:
    # We are likely inside a Jupyter notebook
    from IPython.core.debugger import Tracer


def {{ stencil_name }}_debug(
        {%- for i in stencil_args -%}
        {{ i }}
            {%- if not loop.last -%}
            , {% endif -%}
        {%- endfor -%}):

    try: ipdb.set_trace()
    except NameError: Tracer()()  # Cavalry's here!

    {% if temps -%}
    #
    # temporary data fields to share data among the different stages
    #
    {% for t in temps -%}
    {{ t }} = np.zeros({{ temps_shape }})
    {% endfor -%}
    {%- endif -%}

    {% for stg in stages %}
    for i in range({{ stg["i_start"] }}, {{ stg["i_end"] }}):
        {%- if stg["ndim"] > 1 %}
        for j in range({{ stg["j_start"] }}, {{ stg["j_end"] }}):
        {% endif -%}
            {%- if stg["ndim"] > 2 %}
            for k in range({{ stg["k_start"] }}, {{ stg["k_end"] }}):
            {% endif -%}
        {% for exp in stg["expressions"] -%}
        {% if stg["ndim"] > 1 %}    {% if stg["ndim"] > 2 %}    {% endif -%}{% endif -%}{{ exp }}
        {% endfor -%}
    {% endfor -%}
