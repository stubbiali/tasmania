/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
/**
 * This code was automatically generated by gridtools4py:
 * the Python interface to the Gridtools library
 *
 */
#include <stencil-composition/stencil-composition.hpp>

#include "{{ stg_hdr_file }}"



#ifdef __CUDACC__
#define BACKEND backend<Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend<Host, GRIDBACKEND, Block >
#else
#define BACKEND backend<Host, GRIDBACKEND, Naive >
#endif
#endif



using gridtools::level;
using gridtools::extent;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;



//
// function prototype called from Python (no mangling)
//
extern "C"
{
    void run_{{ stencil_name }} (uint_t dim1, uint_t dim2, uint_t dim3,
                      {%- for p in params %}
                      float_type *{{ p.name }}_buff
                          {%- if not loop.last -%}
                          ,
                          {%- endif -%}
                      {% endfor -%});
}


//
// definition of the special regions in the vertical (k) direction
//
{% for stg in stages[stencil_name] -%}
{% for vr in stg.vertical_regions -%}
typedef gridtools::interval<level<{{ vr.start_splitter }},-1>, level<{{ vr.end_splitter }},-2> > {{ vr.name }};
{% endfor -%}
{% endfor -%}
typedef gridtools::interval<level<0,-2>, level<{{ splitters|length-1 }},1> > axis;



void run_{{ stencil_name }} (uint_t d1, uint_t d2, uint_t d3,
                      {%- for p in params %}
                      float_type *{{ p.name }}_buff
                          {%- if not loop.last -%}
                          ,
                          {%- endif -%}
                      {% endfor -%})
{
    //
    // C-like memory layout
    //
    typedef gridtools::layout_map<0,1,2> layout_t;

    //
    // define the storage unit used by the backend
    //
    typedef meta_storage<meta_storage_aligned<meta_storage_base<__COUNTER__, layout_t, false>, aligned<0>, halo<0,0,0> > > meta_data_t;
    typedef gridtools::BACKEND::storage_type<float_type,
                                             meta_data_t >::type storage_type;

    {% if temps %}
    //
    // define a special data type for the temporary, i.e., intermediate buffers
    //
    typedef gridtools::BACKEND::temporary_storage_type<float_type,
                                                       meta_data_t >::type tmp_storage_type;
    {% endif -%}

    {% if params %}
    //
    // parameter data fields use the memory buffers received from NumPy arrays
    //
    typename storage_type::storage_info_type meta_(d1, d2, d3);

    {% for p in params -%}
    storage_type {{ p.name }} (meta_,
                         (float_type *) {{ p.name }}_buff,
                          "{{ p.name }}");
    {% endfor %}
    {% endif -%}

    //
    // place-holder definition: their order matches the stencil parameters,
    // especially the non-temporary ones, during the construction of the domain
    //
    {% for p in params_temps -%}
    typedef arg<{{ loop.index0 }},
        {%- if scope.is_temporary (p.name) -%}
            tmp_storage_type>
        {%- else -%}
            storage_type>
        {%- endif %} p_{{ p.name|replace('.', '_') }};
    {% endfor %}

    //
    // an array of placeholders to be passed to the domain
    //
    typedef boost::mpl::vector<
        {{- params_temps|join_with_prefix ('p_', attribute='name')|join (', ')|replace('.', '_') }}> arg_type_list;

    //
    // construction of the domain.
    // The domain is the physical domain of the problem, with all the physical
    // fields that are used, temporary and not.
    // It must be noted that the only fields to be passed to the constructor
    // are the non-temporary. The order in which they have to be passed is the
    // order in which they appear scanning the placeholders in order.
    // (I don't particularly like this)
    //
    gridtools::aggregator_type<arg_type_list> domain (boost::fusion::make_vector (
        {{- params|join_with_prefix('&', attribute='name')|join(', ') }}));

    {% for s in stencils %}
    //
    // definition of the physical dimensions of the problem.
    // The constructor takes the horizontal plane dimensions, i.e.:
    //
    // { halo in negative direction,
    //   halo in positive direction,
    //   index of the first interior element,
    //   index of the last interior element,
    //   total number of elements in dimension }
    //
    uint_t di_{{ loop.index0 }}[5] = { {{ s.get_halo()[0] }},
                     {{ s.get_halo()[1] }},
                     {{ s.get_halo()[0] }},
                     d1-{{ s.get_halo()[1] }}-1,
                     d1 };
    uint_t dj_{{ loop.index0 }}[5] = { {{ s.get_halo()[2] }},
                     {{ s.get_halo()[3] }},
                     {{ s.get_halo()[2] }},
                     d2-{{ s.get_halo()[3] }}-1,
                     d2 };

    //
    // the vertical dimension of the problem is a property of this object
    //
    gridtools::grid<axis> grid_{{ loop.index0 }}(di_{{ loop.index0 }}, dj_{{ loop.index0 }});
    {% set grid_id = loop.index0 -%}
    {% for spl in splitters|dictsort -%}
    grid_{{ grid_id}}.value_list[{{ spl[1] }}] = {{ spl[0] }};
    {% endfor -%}

    //
    // Here we do a lot of stuff
    //
    // 1) we pass to the intermediate representation ::run function the
    // description of the stencil, which is a multistage stencil;
    // 2) the logical physical domain with the fields to use;
    // 3) the actual domain dimensions
    //
#ifdef CXX11_ENABLED
    auto
#else
#ifdef __CUDACC__
    gridtools::stencil*
#else
    boost::shared_ptr<gridtools::stencil>
#endif
#endif
    {% set inside_independent_block = False %}

    comp_{{ s.name|lower }} =
      gridtools::make_computation<gridtools::BACKEND>
      (
          domain, grid_{{ loop.index0 }},
            gridtools::make_multistage
            (
                execute<{{ s.get_k_direction() }}>(),
                {% for f in stages[s.name] -%}
                    {% if f.independent and not inside_independent_block -%}
                        gridtools::make_independent (
                        {% set inside_independent_block = True -%}
                    {% endif -%}
                    {% if not f.independent and inside_independent_block -%}
                        ),
                        {% set inside_independent_block = False -%}
                    {% endif -%}
                    gridtools::make_stage<{{ f.name }}>(
                       {{- f.scope.get_parameters ( )|join_with_prefix ('p_', attribute='name')|join ('(), ')|replace('.', '_') }}() )
                       {%- if not (loop.index0 in independent_stage_idx or loop.last) -%}
                       ,
                       {%- endif %}
                {% endfor -%}
            )
      );
    {% endfor %}

    //
    // preparation ...
    //
    {% for s in stencils -%}
    comp_{{ s.name|lower }}->ready();
    {% endfor %}
    {% for s in stencils -%}
    comp_{{ s.name|lower }}->steady();
    {% endfor %}
    //
    // ... and execution
    //
    {% for s in stencils -%}
    comp_{{ s.name|lower }}->run();
    {% endfor %}
    //
    // clean everything up
    //
    {% for s in stencils -%}
    comp_{{ s.name|lower }}->finalize();
    {% endfor %}
}




/**
 * A MAIN function for debugging purposes
 *
int main (int argc, char **argv)
{
    uint_t dim1 = 64;
    uint_t dim2 = 64;
    uint_t dim3 = 32;

    {% for p in params -%}
    float_type *{{ p.name }}_buff = (float_type *) malloc (dim1*dim2*dim3 * sizeof (float_type));
    {% endfor -%}


    // initialization
    for (int i = 0; i<dim1; i++) {
        for (int j = 0; j<dim2; j++) {
            for (int k = 0; k<dim3; k++) {
            {% for p in params -%}
                {{ p.name }}_buff[i*dim3*dim2 + j*dim3 + k] = i*dim3*dim2 + j*dim3 + k;
            {% endfor -%}
            }
        }
    }

    // execution
    run_{{ stencil_name }} (dim1, dim2, dim3,
          {%- for p in params %}
          {{ p.name }}_buff
              {%- if not loop.last -%}
              ,
              {%- endif -%}
          {% endfor -%});

    // output
    for (int i = 0; i<dim1; i++) {
        for (int j = 0; j<dim2; j++) {
            for (int k = 0; k<dim3; k++) {
                    printf ("(%d,%d,%d)", i,j,k);
                {% for p in params -%}
                    printf ("\t%.5f", {{ p.name }}_buff[i*dim3*dim2 + j*dim3 + k]);
                {% endfor -%}
                    printf ("\n");
            }
            }
    }


    return EXIT_SUCCESS;
}
*/
