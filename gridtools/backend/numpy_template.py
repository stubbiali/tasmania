import numpy as np


def {{ stencil_name }}_numpy(
		{%- for i in stencil_args -%}
		{{ i }}
			{%- if not loop.last -%}
			, {% endif -%}
		{%- endfor -%}):
	{% if temps -%}
	#
	# Temporary data fields to share data among the different stages
	#
	{% for t in temps -%}
	{{ t }} = np.zeros({{ temps_shape }})
	{% endfor -%}
	{% endif %}
	#
	# Perform computations
	#
	{%- for stg in stages %}
	{% if stg["vertical_mode"] == "vectorized" -%}
	{{ stg["expression"] }}
	{% else -%}
	for k in range({{ stg["k_start"] }}, {{ stg["k_stop"]}}
		{%- if stg["k_start"] > stg["k_stop"] -%}
		, -1 {%- endif -%}):
		{{ stg["expression"] }}
	{%- endif -%}
	{% endfor -%}
