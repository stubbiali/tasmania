Introduction
============

Background And Motivation
-------------------------

Weather and climate models are complex systems comprising several subsystems
(atmosphere, ocean, land, glacier, sea ice, and marine biogeochemistry) which
interact at their interfaces through the exchange of mass, momentum and energy.
Each domain hosts a plethora of interlinked physical and chemical processes which
can be characterized on a wide spectrum of spatio-temporal scales. Due to the
limited computational resources available, creating a discrete model covering
the entire range of scales in a homogeneous fashion is challenging, if not an
impossible task. This applies to the individual subsystems too. In atmospheric
modeling, an efficient and well-established treatment of the multi-scale nature
of the atmosphere is accomplished by distinguishing between (i) fully resolved
fluid-dynamics features and (ii) subgrid-scale processes. The latter do not
emerge naturally on the computational mesh and form the so-called **physics** of
the model, as opposed to the large-scale **dynamics**.

In a numerical weather prediction (NWP) model, the solution of the underlying
governing equations is delegated to a fluid-dynamics solver called **dynamical core**
(often shortened to **dycore**), whereas **physical parameterizations** express
the bulk effect of the subgrid-scale phenomena upon the large-scale flow. The
procedure which molds all the dynamics and physics components to yield a coherent
and comprehensive model takes the name of **physics-dynamics coupling** (PDC).

The continual growth in model resolution demands an increasing specialization to
address the physical processes which emerge on smaller and smaller scales. This
has resulted in a high compartmentalization of the model development, with
dynamical cores and physics packages mostly developed in isolation. Besides
easing the proliferation of software components with incompatible structure,
such an approach is in direct contrast with the need of improving the time
stepping in the current apparatus of atmospheric models. Indeed as the underlying
mesh is refined, the truncation associated with the single model components
decrease, and the coupling error eventually dominates. More generally, the choice
of time integration appears as prominent as the choice of spatial discretization
at high resolution. This is even more true when using long timesteps as permitted
by semi-implicit semi-Lagrangian schemes. Splitting schemes remain the preferred
option to couple model components in production codes. Given their capability to
treat components in isolation, these schemes allow for a modular code design,
thus facilitating code readability and maintanability.

Tasmania As A Framework
-----------------------

Legacy codes may suffer from limitations in flexibility, readability and
inter-operatility which make them loosely fit the numerical investigation of the
physics-dynamics coupling. The primary goal of Tasmania is providing a more
favorable environment for this kind of research. This is accomplished by
specializing and extending the toolset of infrastructure code offered by
`Sympl <https://sympl.readthedocs.io/en/latest/>`_. This toolset comprises basic
functions and objects which could be used by any type of Earth System models.

Following Sympl, Tasmania conceives a model as an ensemble of components, each
handling a specific dynamics or physics process. This is at contrast with many
frameworks out there, whose components represent entire subdomains (e.g.
atmosphere, ocean) and processes within subdomains are hardly accessible.
Our components act on and interact through the **state**, i.e. the set of
variables which describe the configuration of the physical system underneath at any point
in time. The grid values of a variable (i.e. its **field**) are stored in a
bespoken version of `Xarray <http://xarray.pydata.org/en/stable/index.html>`_'s
:class:`~xarray.DataArray`, and the state is encoded as a dictionary of these
data structures. The chaining of the components is accomplished by the
**couplers**, which pursue a well-defined coupling strategy.

Tasmania As A Library
---------------------

As proof-of-principles, the following concrete models have been implemented on
top of the abstractions introduced above:

- The two-dimensional viscid Burgers' equations;
- A three-dimensional mountain flow model in isentropic coordinates.



References
----------
