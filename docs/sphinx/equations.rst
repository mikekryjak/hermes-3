.. _sec-equations:

Equations
==========

This section contains components which determine which equations are
being solved in the code. There are two broad classes of components:

Whole equations
   For example, ``fixed_temperature``, ``evolve_pressure``, 
   ``evolve_energy`` allow the solution of energy in three levels
   of fidelity: constant temperature, a pressure equation and the
   conservative total energy equation. ``neutral_mixed`` contains
   both parallel and perpendicular transport of neutrals and 
   has several equations included within.

Terms
   For example, ``anomalous_diffusion`` adds cross-field transport
   to the density, energy and momentum equations if they are available
   while  ``diamagnetic_drift`` and ``polarisation_drift`` 
   add drift terms.
   
Please refer to the examples for common component configurations.


Species density
---------------

The density of a species can be calculated in several different ways,
and are usually needed by other components.

.. _fixed_density:

fixed_density
~~~~~~~~~~~~~

Set the density to a value which does not change in time. For example:

.. code-block:: ini

   [d]
   type = fixed_density, ...

   AA = 2 # Atomic mass
   charge = 0
   density = 1e17 # In m^-3

Note that the density can be a function of `x`, `y` and `z` coordinates
for spatial variation.

The implementation is in the `FixedDensity` class:

.. doxygenstruct:: FixedDensity
   :members:

.. _evolve_density:

evolve_density
~~~~~~~~~~~~~~

This component evolves the species density in time, using the BOUT++
time integration solver. The species charge and atomic mass must be set,
and the initial density should be specified in its own section:

.. code-block:: ini

   [d]
   type = evolve_density, ...

   AA = 2 # Atomic mass
   charge = 0

   [Nd]
   function = 1 - 0.5x # Initial condition, normalised to Nnorm

The equation solved is:

.. math::

   \frac{\partial n}{\partial t} = -\nabla\cdot\left[n \left(\frac{1}{B}\mathbf{b}\times\nabla\phi + v_{||}\mathbf{b}\right)\right] + S_n

where the source :math:`S_n` is a combination of external source, and
other processes that nay be included, including drift terms
(e.g. magnetic drift) or atomic processes (e.g. ionization).

Notes:

1. The density will be saved in the output file as `N` + species
   label, e.g `Nd` in the above example.
2. If `diagnose=true` is set in the species options then the net
   source :math:`S_n` is saved as `SN` + species, e.g. `SNd`; the
   external source is saved as `S` + species + `_src` e.g. `Sd_src`.
   The time derivative of density is saved as `ddt(N` + species + `)`
   e.g. `ddt(Nd)`.
3. The density source can be set in the input mesh file as a field
   `S` + species + `_src` e.g. `Sd_src`. This can be overridden by
   specifying the source in the input options.
4. The `poloidal_flows` switch controls whether the X-Y components of
   the ExB flow are included (default is true). If set to `false` then
   ExB flows are only in the X-Z plane.

The implementation is in the `EvolveDensity` class:

.. doxygenstruct:: EvolveDensity
   :members:



fixed_fraction_ions
~~~~~~~~~~~~~~~~~~~

This sets the density of a species to a fraction of the electron density.

.. doxygenstruct:: FixedFractionIons
   :members:

.. _quasineutral:

quasineutral
~~~~~~~~~~~~

This component sets the density of one species, so that the overall
charge density is zero everywhere. This must therefore be done after
all other charged species densities have been calculated. It only
makes sense to use this component for species with a non-zero charge.

.. doxygenstruct:: Quasineutral
   :members:

Species pressure and temperature
--------------------------------

.. _isothermal:

isothermal
~~~~~~~~~~

Sets the temperature of a species to a fixed value which is constant
in space and time. If the species density is set then this component
also calculates the pressure.

By default only saves the temperature once as a non-evolving variable.
If ``diagnose`` is set then pressure is also saved as a time-evolving
variable.

.. code-block:: ini

   [e]
   type = ..., isothermal

   temperature = 10   # Constant temperature [eV]

.. doxygenstruct:: Isothermal
   :members:


fixed_temperature
~~~~~~~~~~~~~~~~~

Sets the temperature of a species to a fixed value which is constant
in time but can vary in space. If the species density is set then this
component also calculates the pressure.

By default only saves the temperature once as a non-evolving variable.
If ``diagnose`` is set then pressure is also saved as a time-evolving
variable.

.. code-block:: ini

   [e]
   type = ..., fixed_temperature

   temperature = 10 - x   # Spatially dependent temperature [eV]

.. doxygenstruct:: FixedTemperature
   :members:

.. _evolve_pressure:

evolve_pressure
~~~~~~~~~~~~~~~

Evolves the pressure in time. This pressure is named `P<species>` where `<species>`
is the short name of the evolving species e.g. `Pe`.

By default parallel thermal conduction is included, which requires a collision
time. If collisions are not calculated, then thermal conduction should be turned off
by setting `thermal_conduction = false` in the input options.

The choice of collision frequency used for conduction is set by the flag `conduction_collisions_mode`: 
`multispecies` uses all available collision frequencies involving the chosen species, while `braginskii` uses only
self-collisions .The default is `multispecies` and it is recommended for use if solving more than one ion.
If you are solving for a single ion and want to recover Braginskii, use the `braginskii` mode.


If the component option ``diagnose = true`` then additional fields
will be saved to the dump files: The species temperature ``T + name``
(e.g. ``Td+`` or ``Te``), the time derivative ``ddt(P + name)``
(e.g. ``ddt(Pd+)`` or ``ddt(Pe)``), and the source of pressure from
other components is saved as ``SP + name`` (e.g. ``SPd+`` or ``SPe``).
The pressure source is the energy density source multiplied by ``2/3``
(i.e. assumes a monatomic species).

.. math::

   \frac{\partial P}{\partial t} = -\nabla\cdot\left(P\mathbf{v}\right) - \frac{2}{3} P \nabla\cdot\mathbf{b}v_{||} + \frac{2}{3}\nabla\cdot\left(\kappa_{||}\mathbf{b}\mathbf{b}\cdot\nabla T\right) + \frac{2}{3}S_E + S_N\frac{1}{2}mNV^2

where :math:`S_E` is the ``energy_source`` (thermal energy source),
and :math:`S_N` is the density source.

Notes:

- Heat conduction through the boundary is turned off currently. This is because
  heat losses are usually calculated at the sheath, so any additional heat conduction
  would be in addition to the sheath heat transmission already included.

The implementation is in `EvolvePressure`:

.. doxygenstruct:: EvolvePressure
   :members:

.. _evolve_energy:

evolve_energy
~~~~~~~~~~~~~

*Note* This is currently under development and has some unresolved
issues with boundary conditions.  Only for testing purposes.

This evolves the sum of species internal energy and parallel kinetic
energy, :math:`\mathcal{E}`:

.. math::

   \mathcal{E} = \frac{1}{\gamma - 1} P + \frac{1}{2}m nv_{||}^2

Note that this component requires the parallel velocity :math:`v_{||}`
to calculate the pressure. It must therefore be listed after a component
that sets the velocity, such as `evolve_momentum`:

.. code-block:: ini

   [d]
   type = ..., evolve_momentum, evolve_energy

The energy density will be saved as `E<species>` (e.g `Ed`) and the
pressure as `P<species>` (e.g. `Pd`). Additional diagnostics, such as the
temperature, can be saved by setting the option `diagnose = true`.

.. doxygenstruct:: EvolveEnergy
   :members:

SNB nonlocal heat flux
~~~~~~~~~~~~~~~~~~~~~~

Calculates the divergence of the electron heat flux using the
Shurtz-Nicolai-Busquet (SNB) model. Uses the BOUT++ implementation which is
`documented here <https://bout-dev.readthedocs.io/en/latest/user_docs/nonlocal.html?#snb-model>`_.

.. doxygenstruct:: SNBConduction
   :members:


simple_conduction
~~~~~~~~~~~~~~~~~~~~~~

This is a simplified parallel heat conduction model that can be used when a linearised model is needed.
If used, the thermal conduction term in `evolve_pressure` component should be disabled.

.. code-block:: ini

   [hermes]
   components = e, ...

   [e]
   type = evolve_pressure, simple_conduction

   thermal_conduction = false  # Disable term in evolve_pressure

To linearise the heat conduction the temperature and density used in
calculating the Coulomb logarithm and heat conduction coefficient can
be fixed by specifying `conduction_temperature` and
`conduction_density`.

Note: For hydrogenic plasmas this produces very similar parallel electron
heat conduction as the `evolve_pressure` term with electron-electron collisions
disabled.

.. doxygenstruct:: SimpleConduction
   :members:


Species parallel dynamics
-------------------------

fixed_velocity
~~~~~~~~~~~~~~

Sets the velocity of a species to a fixed value which is constant
in time but can vary in space. If the species density is set then this
component also calculates the momentum.

Saves the temperature once as a non-evolving variable.

.. code-block:: ini

   [e]
   type = ..., fixed_velocity

   velocity = 10 + sin(z)   # Spatially dependent velocity [m/s]

.. doxygenstruct:: FixedVelocity
   :members:


.. _evolve_momentum:

evolve_momentum
~~~~~~~~~~~~~~~

Evolves the momentum `NV<species>` in time. The evolving quantity includes the atomic
mass number, so should be divided by `AA` to obtain the particle flux.

If the component option ``diagnose = true`` then additional fields
will be saved to the dump files: The velocity ``V + name``
(e.g. ``Vd+`` or ``Ve``), the time derivative ``ddt(NV + name)``
(e.g. ``ddt(NVd+)`` or ``ddt(NVe)``), and the source of momentum
density (i.e force density) from other components is saved as ``SNV +
name`` (e.g. ``SNVd+`` or ``SNVe``).

The implementation is in ``EvolveMomentum``:

.. doxygenstruct:: EvolveMomentum
   :members:


.. _zero_current:

zero_current
~~~~~~~~~~~~

This calculates the parallel flow of one charged species so that there is no net current,
using flows already calculated for other species. It is used like `quasineutral`:

.. code-block:: ini

   [hermes]
   components = h+, ..., e, ...   # Note: e after all other species
   
   [e]
   type = ..., zero_current,... # Set e:velocity

   charge = -1 # Species must have a charge

.. doxygenstruct:: ZeroCurrent
   :members:

electron_force_balance
~~~~~~~~~~~~~~~~~~~~~~

This calculates a parallel electric field which balances the electron
pressure gradient and other forces on the electrons (including
collisional friction, thermal forces):

.. math::

   E_{||} = \left(-\nabla p_e + F\right) / n_e

where :math:`F` is the `momentum_source` for the electrons.
This electric field is then used to calculate a force on the other species:

.. math::

   F_z = Z n_z E_{||}

which is added to the ion's `momentum_source`. 

The implementation is in `ElectronForceBalance`:

.. doxygenstruct:: ElectronForceBalance
   :members:

.. _electron_viscosity:

electron_viscosity
~~~~~~~~~~~~~~~~~~~~~~

Calculates the Braginskii electron parallel viscosity, adding a force (momentum source)
to the electron momentum equation:

.. math::

   F = \sqrt{B}\nabla\cdot\left[\frac{\eta_e}{B}\mathbf{b}\mathbf{b}\cdot\nabla\left(\sqrt{B}V_{||e}\right)\right]

The electron parallel viscosity is

.. math::

   \eta_e = \frac{4}{3} 0.73 p_e \tau_e

where :math:`\tau_e` is the electron collision time. The collisions between electrons
and all other species therefore need to be calculated before this component is run:

.. code-block:: ini

   [hermes]
   components = ..., e, ..., collisions, electron_viscosity

.. doxygenstruct:: ElectronViscosity
   :members:

.. _ion_viscosity:

ion_viscosity
~~~~~~~~~~~~~~~~~~~~~~

Adds ion viscosity terms to all charged species that are not electrons.
The collision frequency is required so this is a top-level component that
must be calculated after collisions:

.. code-block:: ini

   [hermes]
   components =  ..., collisions, ion_viscosity

By default only the parallel diffusion of momentum is included, adding a force to each
ion's momentum equation:

.. math::

   F = \sqrt{B}\nabla\cdot\left[\frac{\eta_i}{B}\mathbf{b}\mathbf{b}\cdot\nabla\left(\sqrt{B}V_{||i}\right)\right]

The ion parallel viscosity is

.. math::

   \eta_i = \frac{4}{3} 0.96 p_i \tau_i

The choice of collision frequency is set by the flag `viscosity_collisions_mode`: `multispecies` uses
all available collision frequencies involving the chosen species, while `braginskii` uses only
ii collisions. The default is `multispecies` and it is recommended when solving
more than one ion. If you are solving for a single ion and want to recover Braginskii, 
use the `braginskii` mode.


If the `perpendicular` option is set:

.. code-block:: ini

   [ion_viscosity]
   perpendicular = true # Include perpendicular flows

Then the ion scalar viscous pressure is calculated as:

.. math::

   \Pi_{ci} = \Pi_{ci||} + \Pi_{ci\perp}

where :math:`\Pi_{ci||}` corresponds to the parallel diffusion of momentum above.

.. math::

   \Pi_{ci||} = - 0.96 \frac{2p_i\tau_i}{\sqrt{B}} \partial_{||}\left(\sqrt{B} V_{||i}\right)

The perpendicular part is calculated from:

.. math::

   \begin{aligned}\Pi_{ci\perp} =& 0.96 p_i\tau_i \kappa \cdot \left[\mathbf{V}_E + \mathbf{V}_{di} + 1.16\frac{\mathbf{b}\times\nabla T_i}{B} \right] \\
   =& -0.96 p_i\tau_i\frac{1}{B}\left(\mathbf{b}\times\kappa\right)\cdot\left[\nabla\phi + \frac{\nabla p_i}{en_i} + 1.61\nabla T_i \right]\end{aligned}


A parallel force term is added, in addition to the parallel viscosity above:

.. math::

   F = -\frac{2}{3}B^{3/2}\partial_{||}\left(\frac{\Pi_{ci\perp}}{B^{3/2}}\right)
   
In the vorticity equation the viscosity appears as a divergence of a current:

.. math::

   \mathbf{J}_{ci} = \frac{\Pi_{ci}}{2}\nabla\times\frac{\mathbf{b}}{B} - \frac{1}{3}\frac{\mathbf{b}\times\nabla\Pi_{ci}}{B}

that transfers energy between ion internal energy and :math:`E\times B` energy:

.. math::

   \begin{aligned}\frac{\partial \omega}{\partial t} =& \ldots + \nabla\cdot\mathbf{J}_{ci} \\
   \frac{\partial p_i}{\partial t} =& \ldots - \mathbf{J}_{ci}\cdot\nabla\left(\phi + \frac{p_i}{n_0}\right)\end{aligned}

Note that the sum of the perpendicular and parallel contributions to the ion viscosity act to damp
the net poloidal flow. This can be seen by assuming that :math:`\phi`, :math:`p_i` and :math:`T_i`
are flux functions. We can then write:

.. math::

   \Pi_{ci\perp} = -0.96 p_i\tau_i \frac{1}{B}\left(\mathbf{b}\times\kappa\right)\cdot\nabla\psi F\left(\psi\right)

where

.. math::

   F\left(\psi\right) = \frac{\partial\phi}{\partial\psi} + \frac{1}{en}\frac{\partial p_i}{\partial\psi} + 1.61\frac{\partial T_i}{\partial\psi}

Using the approximation

.. math::

   \left(\mathbf{b}\times\kappa\right)\cdot\nabla\psi \simeq -RB_\zeta \partial_{||}\ln B

expanding:

.. math::

   \frac{2}{\sqrt{B}}\partial_{||}\left(\sqrt{B}V_{||i}\right) = 2\partial_{||}V_{||i} + V_{||i}\partial_{||}\ln B

and neglecting parallel gradients of velocity gives:

.. math::

   \Pi_{ci} \simeq 0.96 p_i\tau_i \left[ \frac{RB_{\zeta}}{B}F\left(\psi\right) - V_{||i} \right]\partial_{||}\ln B

   
**Notes** and implementation details:
- The magnitude of :math:`\Pi_{ci\perp}` and :math:`\Pi_{ci||}` are individually
  limited to be less than or equal to the scalar pressure :math:`Pi` (though can have
  opposite sign). The reasoning is that if these off-diagonal terms become large then
  the model is likely breaking down. Occasionally happens in low-density regions.

   
.. doxygenstruct:: IonViscosity
   :members:

.. _thermal_force:

thermal_force
~~~~~~~~~~~~~

This implements simple expressions for the thermal force. If the
`electron_ion` option is true (which is the default), then a momentum
source is added to all ions:

.. math::

   F_z = 0.71 n_z Z^2 \nabla_{||}T_e

where :math:`n_z` is the density of the ions of charge :math:`Z`. There
is an equal and opposite force on the electrons.

If the `ion_ion` option is true (the default), then forces are
calculated between light species (atomic mass < 4) and heavy species
(atomic mass > 10).  If any combinations of ions are omitted, then a
warning will be printed once.
The force on the heavy ion is:

.. math::

   \begin{aligned}
   F_z =& \beta \nabla_{||}T_i \\
   \beta =& \frac{3\left(\mu + 5\sqrt{2}Z^2\left(1.1\mu^{5/2} - 0.35\mu^{3/2}\right) - 1\right)}{2.6 - 2\mu + 5.4\mu^2} \\
   \mu =& m_z / \left(m_z + m_i\right)
   \end{aligned}

where subscripts :math:`z` refer to the heavy ion, and :math:`i`
refers to the light ion. The force on the light ion fluid is equal and
opposite: :math:`F_i = -F_z`.

The implementation is in the `ThermalForce` class:

.. doxygenstruct:: ThermalForce
   :members:


Neutral gas models
------------------

In 1D, neutral transport is currently done through the same components as for plasma, i.e. `evolve_density`,
`evolve_momentum` and `evolve_pressure` with the additional, optional `neutral_parallel_diffusion` component.
In 2D, all of this functionality is implemented in one component called `neutral_mixed` which additionally
has cross-field transport. This discrepancy is due to historical reasons and will be refactored.


.. _neutral_parallel_diffusion:

1D: neutral_parallel_diffusion
~~~~~~~~~~~~~~~~~~~~~~~~~~

This adds diffusion to **all** neutral species (those with no or zero charge),
because it needs to be calculated after the collision frequencies are known.

.. code-block:: ini

   [hermes]
   components = ... , collisions, neutral_parallel_diffusion

   [neutral_parallel_diffusion]
   dneut = 1         # Diffusion multiplication factor
   diagnose = true   # This enables diagnostic output for each species


It is intended mainly for 1D simulations, to provide effective parallel
diffusion of particles, momentum and energy due to the projection of
cross-field diffusion:

.. math::

   \begin{aligned}
   \frac{\partial n_n}{\partial t} =& \ldots + \nabla\cdot\left(\mathbf{b}D_n n_n \frac{1}{p_n}{\partial_{||}p_n}\right) \\
   \frac{\partial p_n}{\partial t} =& \ldots + \nabla\cdot\left(\mathbf{b}D_n p_n \frac{1}{p_n}\partial_{||}p_n\right) + \frac{2}{3}\nabla\cdot\left(\mathbf{b}\kappa_n \partial_{||}T_n\right) \\
   \frac{\partial}{\partial t}\left(m_nn_nv_{||n}\right) =& \ldots + \nabla\cdot\left(\mathbf{b}D_n m_n n_nv_{||n} \frac{1}{p_n} \partial_{||}p_n\right) + \nabla\cdot\left(\mathbf{b}\eta_n \partial_{||}v_{||n}\right)
   \end{aligned}

The diffusion coefficient is in :math:`m^2/s` and is calculated as

.. math::

   D_n = \left(\frac{B}{B_{pol}}\right)^2 \frac{eT_n}{m_{n} \nu}

where `m_{n}` is the neutral species mass in kg and :math:`\nu` is the collision
frequency (by default, this sums up all of the enabled neutral collisions from 
the collisions component as well as the charge exchange rate).
The factor :math:`B / B_{pol}` is the projection of the cross-field
direction on the parallel transport, and is the `dneut` input setting. Currently, the recommended
use case for this component is to represent the neutrals diffusing orthogonal to the target wall, and
it is recommended to set `dneut` according to the field line pitch at the target.

.. doxygenstruct:: NeutralParallelDiffusion
   :members:

.. _neutral_mixed:

2D/3D: neutral_mixed
~~~~~~~~~~~~~~~~~~~~~~~~~~


The below describes the `neutral_mixed` component used for 2D and 3D simulations. Note that all dimensionalities
are compatible with the `neutral_boundary` component which facilitates energy losses to the wall through neutral reflection.

The `neutral_mixed` component solves fluid equations along :math:`y`
(parallel to the magnetic field), and uses diffusive transport in :math:`x`
and :math:`z`.  It was adopted from the approach used in UEDGE and this [M.V. Umansky, J.N.M (2003)]. The Hermes-3 approach
is more advanced in having a separate neutral pressure equation, similar to the 
new AFN (Advanced Fluid Neutral) model in SOLPS-ITER [N. Horsten, N.F. (2017)].

.. math::
   
   \begin{aligned}

   \frac{\partial n_n}{\partial t} =& -\nabla\cdot\left(n_n\mathbf{b}v_{||n} + n_n\mathbf{v}_{\perp n}\right) \\
         &    + S \\
   \frac{\partial}{\partial t}\left(n_nv_{||n}\right) =& -\nabla\cdot\left(n_nv_{||n} \mathbf{b}v_{||n} + n_nv_{||n}\mathbf{v}_{\perp n}\right) \\
         &    - \partial_{||}p_n \\
         &    + \nabla \cdot (m_n \eta_{n} \nabla_{\perp} v_{\parallel n}) + \nabla \cdot( m_n \eta_{n} \nabla{\parallel} v_{\parallel n} ) \\
         &    + F \\
   \frac{\partial p_n}{\partial t} =& -\nabla\cdot\left(p_n\mathbf{b}v_{||n} + \frac{5}{3} p_n\mathbf{v}_{\perp n}\right) \\
         &    - \frac{2}{3}p_n\nabla\cdot\left(\mathbf{b}v_{||n}\right) \\
         &    + \frac{2}{3} \nabla\cdot\left(\kappa_n \nabla_\perp T_n\right) + \frac{2}{3} \nabla\cdot\left(\kappa_n \nabla_{\parallel} T_n\right) \\
         &    - \frac{2}{3} v_n \nabla \cdot (m_n \eta_{n} \nabla_{\perp} v_{\parallel n}) + \frac{2}{3} \nabla \cdot( m_n \eta_{n} \nabla_{\parallel} v_{\parallel n} ) \\
         &    + \frac{2}{3}E \\

   \end{aligned}

Where for the density equation, the first row of terms contains the parallel and perpendicular 
advection and the second row the particle sources. In the parallel momentum equation, the first row of terms
features parallel and perpendicular advection of parallel momentum. This is followed by the compression term
and the perpendicular and parallel viscosity (diffusion of parallel momentum) as well as the momentum source term.
In the pressure equation, the first row contains the parallel and perpendicular advection of pressure. This is followed
by the compression term, the perpendicular and parallel conduction (diffusion of temperature) and perpendicular and parallel
viscous heating, finally followed by the energy sources.

While parallel momentum is evolved and is exchanged with the plasma parallel momentum, the advection of momentum is neglected in the perpendicular direction,
resulting in the pressure diffusion model, where the pressure gradient is balanced by frictional forces. This is similar to Fickian diffusion with the pressure
gradient replacing the density gradient as the flow driver, in an approach similar to that taken in nuclear fission neutronic transport modelling and several other edge codes.

The perpendicular velocity is calculated as:

.. math::
   \begin{aligned}
   v_{\perp} =& -D_n \frac{1}{P_n} \nabla_{\perp} p_n
   \end{aligned}

Where in the code, :math:`\frac{1}{P_n} \nabla_{\perp}P_n` is represented as :math:`ln(P_n)`, which helps
preserve pressure positivity. 

The choice of collision frequency is set by the flag `diffusion_collisions_mode`: `multispecies` uses
all available collision frequencies involving the chosen species, while `afn` uses only
CX and IZ rates. The default is `afn` and corresponds to the choice in UEDGE and 
the SOLPS-ITER AFN (Advanced Fluid Neutral) model. 

The diffusion coefficients are defined as:

.. math::

   \begin{aligned} 
   D_n =& v_{th,n}^{2} \nu_{n, tot}  \\
   \kappa_{n} =& \frac{5}{2} D_n N_n \\
   \eta_{n} =& \frac{2}{5} m_n \kappa_{n} \\
   \end{aligned}

Where :math:`v_{th,n}= \sqrt{\frac{T_n}{m_n}}` is the thermal velocity of neutrals and :math:`\nu_{n, tot}` is the total
neutral collisionality. This is primarily driven by charge exchange and ionisation, which can cause issues in regions
where plasma density is low. Because of this, an additional pseudo-collisionality is calculated based on the maximum vessel 
mean free path and added to the total neutral collisionality.

In an additional effort to limit the diffusivitiy to more physical values, a flux limiter has been implemented which clamps
:math:`D_n` to :math:`D_{n,max}` defined as:

.. math::

   \begin{aligned}
   D_{n,max} =& f_l \frac{v_{th,n}}{abs(\nabla ln(P_n) + 1/l_{max}}
   \end{aligned}

This formulation is equivalent to defining a :math:`D_n` with a free streaming velocity while accounting for the pseudo collisionality due 
to the maximum vessel mean free path :math:`l_{max}`. The flux limiter :math:`f_l` is set to 1.0 by default.

.. doxygenstruct:: NeutralMixed
   :members:

2D/3D: neutral_full_velocity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This model evolves the equations for a neutral fluid, assuming
axisymmetry (constant in :math:`Z`), for the density :math:`n_n`,
velocity :math:`\mathbf{v}_n` and pressure :math:`p_n`.

.. math::

   \begin{aligned}
   \frac{\partial n_n}{\partial t} =& -\nabla\cdot\left(n_n\mathbf{v}_n\right) \nonumber \\
   \frac{\partial \mathbf{v}_n}{\partial t} =& - \mathbf{v}_n\cdot\nabla\mathbf{v}_n -\frac{1}{n_n}\nabla p_n + \frac{1}{n_n}\nabla\cdot\left(\mu \nabla\mathbf{v}\right) + \nabla\cdot\left(\nu \nabla \mathbf{v}_n\right) \\
   \frac{\partial p_n}{\partial t} =& -\gamma \nabla\cdot\left(p_n\mathbf{v}_n\right) + \left(\gamma - 1\right)\mathbf{v}_n\cdot\nabla p_n + \nabla\cdot\left(n_n \chi_n \nabla T_n\right) \nonumber
   \end{aligned}

where the adiabatic index :math:`\gamma` and dissipation parameters
:math:`\nu` (kinematic viscosity) and :math:`\chi` (thermal
conduction) are constants set in the options:

.. code-block:: ini

   [d]
   type = neutral_full_velocity

   adiabatic_index = 5./3 # Ratio of specific heats
   viscosity = 1.0   # Kinematic viscosity [m^2/s]
   conduction = 1.0  # Heat conduction [m^2/s]

The contravariant components of :math:`\mathbf{v}_n` are evolved in
the same :math:`\left(x,y,z\right)` field-aligned coordinate system as
the plasma.  To evaluate the nonlinear advection term, whilst avoiding
the use of noisy Christoffel symbols coming from derivatives of basis
vectors, these components are transformed into
:math:`\left(R,Z,\phi\right)` cylindrical coordinates, advected, then
transformed back. This is done using matrices which are calculated in
the initialisation stage by finite differences of the input mesh:

.. math::

   \begin{aligned}
   \left(\begin{array}{c}
   \nabla R \\
   \nabla Z\end{array}\right) =& \left(\begin{array}{cc}
   \frac{\partial R}{\partial x} & \frac{\partial R}{\partial y} \\
   \frac{\partial Z}{\partial x} & \frac{\partial Z}{\partial y}\end{array}\right)\left(\begin{array}{c}
   \nabla x \\
   \nabla y\end{array}\right) \\
   =& \left(\begin{array}{cc}
   \texttt{Urx} & \texttt{Ury} \\
   \texttt{Uzx} & \texttt{Uzy} \end{array}\right)\left(\begin{array}{c}
   \nabla x \\
   \nabla y\end{array}\right)
   \end{aligned}

These components are calculated by finite differences of the `Rxy` and
`Zxy` arrays in the input, then adjusted to match the given values of
`hthe` and `Bpxy`:

.. math::

   \sqrt{\left(\frac{\partial R}{\partial y}\right)^2 + \left(\frac{\partial R}{\partial y}\right)^2} = h_\theta


.. math::

   \sqrt{\left(\frac{\partial R}{\partial x}\right)^2 + \left(\frac{\partial R}{\partial x}\right)^2} = 1 / \left(R B_\theta\right)


(Note that this second equality only works if :math:`x` and :math:`y`
are orthogonal).

This matrix is then inverted, to give:

.. math::

   \begin{aligned}
   \left(\begin{array}{c}
   \nabla x \\
   \nabla y\end{array}\right) =& \left(\begin{array}{cc}
   \texttt{Txr} & \texttt{Tyr} \\
   \texttt{Txz} & \texttt{Tyz} \end{array}\right)\left(\begin{array}{c}
   \nabla R \\
   \nabla Z\end{array}\right)
   \end{aligned}

The components of :math:`\mathbf{v}_n` are evolved in contravariant form:

.. math::

   \mathbf{v}_n = v^x \mathbf{e}_x + v^y \mathbf{e}_y + v^z \mathbf{e}_z

These components are stored in the output. In the RHS function the
velocity is converted to covariant form:

.. math::

   \mathbf{v}_n = v_x \nabla x + v_y \nabla y + v_z \nabla z

which is then transformed to :math:`v_r`, :math:`v_Z` and :math:`v_\phi`:

.. math::

   \begin{aligned}
   v_r =& \mathbf{v}_n \cdot \nabla R = \frac{\partial x}{\partial R} v_x + \frac{\partial y}{\partial R} v_y \\
   v_Z =& \mathbf{v}_n \cdot \nabla Z = \frac{\partial x}{\partial Z} v_x +  \frac{\partial y}{\partial Z} v_y \\
   v_\phi =& \mathbf{v}_n \cdot \hat{\phi} = v_z / \left(\sigma_{Bpol} R\right)
   \end{aligned}

which are implemented as

.. code-block:: c++

   Field2D vr = Txr * Vn2D.x + Tyr * Vn2D.y;
   Field2D vz = Txz * Vn2D.x + Tyz * Vn2D.y;
   Field2D vphi = Vn2D.z / (sigma_Bp * Rxy);

These components are then advected as scalars for the
:math:`\mathbf{v}_n\cdot\nabla\mathbf{v}_n` term, and are diffused for
the :math:`\nabla\cdot\left(\mu \nabla\mathbf{v}\right)` kinematic
viscosity. 

The advection of momentum :math:`\mathbf{v}\cdot\nabla\mathbf{v}` is
controlled by these settings:

#. `momentum_advection` is `false` by default, disabling this
   nonlinear advection term. This keeps the inertia in the time
   derivative, but neglects the neutral dynamic pressure in the
   momentum balance.
   
#. `toroidal_flow` is `true` by default, which includes the toroidal
   (:math:`z`) component of the neutral flow. Importantly, this allows
   the parallel and poloidal flows to evolve independently: The
   parallel flow can follow the plasma towards the target, while the
   poloidal flow can be away from the target.

#. `curved_torus` is `true` by default, and is only active when both
   `momentum_advection` and `toroidal_flow` are enabled. Neutrals
   travel in straight lines in real space, so toroidal flow is
   converted to radial flow. This appears in the :math:`v_r` and
   :math:`v_\phi` equations due to a combination of the radial
   centrifugal force and conservation of toroidal angular momentum.

Flow perpendicular to the magnetic field is damped by collisions
e.g. CX reactions with the plasma. The steady-state flow is therefore
a balance between the pressure gradient (including dynamic pressure if `momentum_advection` is enabled),
and this friction. The neutral velocity perpendicular to the magnetic field is:

.. math::

   \begin{aligned}
   \mathbf{v}_{n\perp} =& \mathbf{v}_{n} - \mathbf{b}\mathbf{b}\cdot\mathbf{v}_{n} \\
   =& \mathbf{v}_{n} - \mathbf{e}_y\frac{v_{ny}}{g_{yy}} \\
   =& \mathbf{v}_{n} - \left(\nabla y + \frac{g_{yz}}{g_{yy}}\nabla z\right)v_{ny} \\
   =& v_{nx}\nabla x + \left(v_{nz} - \frac{g_{yz}}{g_{yy}}v_{ny}\right)\nabla z
   \end{aligned}

At boundaries neutral thermal energy is lost at a rate controlled by
the option

.. code-block:: ini

   neutral_gamma = 5./4

This sets the flux of power to the wall to:

.. math::

   q = \gamma n_n T_n c_s

Currently this is only done at target boundaries, not radial
boundaries.

Drifts and transport
--------------------

The ExB drift is included in the density, momentum and pressure evolution equations if
potential is calculated. Other drifts can be added with the following components.

diamagnetic_drift
~~~~~~~~~~~~~~~~~

Adds diamagnetic drift terms to all species' density, pressure and parallel momentum
equations. Calculates the diamagnetic drift velocity as

.. math::

   \mathbf{v}_{dia} = \frac{T}{q} \nabla\times\left(\frac{\mathbf{b}}{B}\right)

where the curvature vector :math:`\nabla\times\left(\frac{\mathbf{b}}{B}\right)`
is read from the `bxcv` mesh input variable.

Two forms are available. Form 0 uses the diamagnetic velocity perpendicular to b and the gradient of P; 
at the boundaries this velocity is perpendicular to the boundary. Form 1 uses the magnetic gyro-center drifts, which are mostly vertical;
at the boundaries this form produces a flow through the boundary. 
Forms 0 and 1 are analytically equivalent and should give the same result away from boundaries, 
but form 0 doesn't produce flows through boundaries. This is an approach that UEDGE uses to avoid unphysical boundary flows.


However, Form 1 is nice because the flow velocity depends on the temperature, not the pressure gradient. 
This usually makes it better behaved numerically. To make the most of both, the `diamagnetic_drift` component allows the forms to be mixed
using the ``diamag_form`` setting. For example, the `tcv-x21` example blends it such that form 0 is at the boundary:

.. code-block:: ini

   [diamagnetic_drift]
   diamag_form = x * (1 - x)  # 0 = gradient; 1 = divergence

.. doxygenstruct:: DiamagneticDrift
   :members:


polarisation_drift
~~~~~~~~~~~~~~~~~~

This calculates the polarisation drift of all charged species,
including ions and electrons. It works by approximating the drift
as a potential flow:

.. math::

   \mathbf{v}_{pol} = - \frac{m}{q B^2} \nabla_\perp\phi_{pol}

where :math:`\phi_{pol}` is approximately the time derivative of the
electrostatic potential :math:`\phi` in the frame of the fluid, with
an ion diamagnetic contribution. This is calculated by inverting a
Laplacian equation similar to that solved in the vorticity equation.

This component needs to be run after all other currents have been
calculated.  It marks currents as used, so out-of-order modifications
should raise errors.

See the `examples/blob2d-vpol` example, which contains:

.. code-block:: ini

   [hermes]
   components = e, vorticity, sheath_closure, polarisation_drift

   [polarisation_drift]
   diagnose = true

Setting `diagnose = true` saves `DivJ` to the dump files with the divergence of all
currents except polarisation, and `phi_pol` which is the polarisation flow potential.

.. doxygenstruct:: PolarisationDrift
   :members:

Stellarator cross-field transport: binormal_stpm
~~~~~~~~~~~~~~~~~~~~~~~~~~

This adds a term to **all** species which includes the effects of cross-field
drifts following the stellarator two point model:
`Y. Feng et al., Plasma Phys. Control. Fusion 53 (2011) 024009 <http://dx.doi.org/10.1088/0741-3335/53/2/024009>`_

.. code-block:: ini

   [hermes]
   components = ... , binormal_stpm

   [binormal_stpm]
   D = 1         # [m^2/s]  Density diffusion coefficient
   chi = 3       # [m^2/s]  Thermal diffusion coefficient
   nu = 1        # [m^2/s]  Momentum diffusion coefficient

   Theta = 1e-3  # Field line pitch

It is intended only for 1D simulations, to provide effective parallel
diffusion of particles, momentum and energy due to the projection of
cross-field diffusion:

.. math::

   \begin{aligned}
   \frac{\partial N}{\partial t} =& \ldots + \nabla\cdot\left(\mathbf{b}\frac{D}{\Theta}\partial_{||}N\right) \\
   \frac{\partial P}{\partial t} =& \ldots + \frac{2}{3}\nabla\cdot\left(\mathbf{b}\frac{\chi}{\Theta} N\partial_{||}T\right) \\
   \frac{\partial}{\partial t}\left(NV\right) =& \ldots + \nabla\cdot\left(\mathbf{b}\frac{\nu}{\Theta} \partial_{||}NV\right) 
   \end{aligned}
   
The diffusion coefficients `D`, `\chi` and `\nu` and field line pitch `\Theta` are prescribed in the input file.


.. doxygenstruct:: BinormalSTPM
   :members:


Tokamak cross-field transport: anomalous_diffusion
~~~~~~~~~~~~~~~~~~~

Adds cross-field diffusion of particles, momentum and energy to a species.

.. code-block:: ini

   [hermes]
   components = e, ...

   [e]
   type = evolve_density, evolve_momentum, evolve_pressure, anomalous_diffusion

   anomalous_D = 1.0   # Density diffusion [m^2/s]
   anomalous_chi = 0,5 # Thermal diffusion [m^2/s]
   anomalous_nu = 0.5  # Kinematic viscosity [m^2/s]

Anomalous diffusion coefficients can be functions of `x` and `y`.  The
coefficients can also be read from the mesh input file: If the mesh
file contains `D_` + the name of the species, for example `D_e` then
it will be read and used as the density diffusion coefficient.
Similarly, `chi_e` is the thermal conduction coefficient, and `nu_e`
is the kinematic viscosity. All quantities should be in SI units of
m^2/s.  Values that are set in the options (as above) override those
in the mesh file.

The sources of particles :math:`S`, momentum :math:`F` and energy
:math:`E` are calculated from species density :math:`N`, parallel
velocity :math:`V` and temperature :math:`T` using diffusion
coefficients :math:`D`, :math:`\chi` and :math:`\nu` as follows:

.. math::

   \begin{aligned}
   S =& \nabla\cdot\left(D \nabla_\perp N\right) \\
   F =& \nabla\cdot\left(m V D \nabla_\perp N\right) + \nabla\cdot\left(m N \nu \nabla_\perp V\right)\\
   E =& \nabla\cdot\left(\frac{3}{2}T D \nabla_\perp N\right) + \nabla\cdot\left(N \chi \nabla_\perp T\right)
   \end{aligned}

Note that particle diffusion is treated as a density gradient-driven flow
with velocity :math:`v_D = -D \nabla_\perp N / N`.

.. doxygenstruct:: AnomalousDiffusion
   :members:


Electromagnetic fields
----------------------

These are components which calculate the electric and/or magnetic
fields.

.. _vorticity:

vorticity
~~~~~~~~~

Evolves a vorticity equation, and at each call to transform() uses a matrix
inversion to calculate potential from vorticity.

In this component the Boussinesq approximation is made, so the
vorticity equation solved is

.. math::

   \nabla\cdot\left(\frac{\overline{A}\overline{n}}{B^2}\nabla_\perp \phi\right) \underbrace{+ \nabla\cdot\left(\sum_i\frac{A_i}{Z_i B^2}\nabla_\perp p_i\right)}_{\mathrm{if diamagnetic\_polarisation}} = \Omega

Where the sum is over species, :math:`\overline{A}` is the average ion
atomic number, and :math:`\overline{n}` is the normalisation density
(i.e. goes to 1 in the normalised equations). The ion diamagnetic flow
terms in this Boussinesq approximation can be written in terms of an
effective ion pressure :math:`\hat{p}`:

.. math::

   \hat{p} \equiv \sum_i \frac{A_i}{\overline{A} Z_i} p_i

as

.. math::

   \nabla\cdot\left[\frac{\overline{A}\overline{n}}{B^2}\nabla_\perp \left(\phi + \frac{\hat{p}}{\overline{n}}\right) \right] = \Omega
   
Note that if ``diamagnetic_polarisation = false`` then the ion
pressure terms are removed from the vorticity, and also from other ion
pressure terms coming from the polarisation current
(i.e. :math:`\hat{p}\rightarrow 0`.

This is a simplified version of the full vorticity definition which is:

.. math::

   \nabla\cdot\left(\sum_i \frac{A_i n_i}{B^2}\nabla_\perp \phi + \sum_i \frac{A_i}{Z_i B^2}\nabla_\perp p_i\right) = \Omega

and is derived by replacing

.. math::

   \sum_i A_i n_i \rightarrow \overline{A}\overline{n}

In the case of multiple species, this Boussinesq approximation means that the ion diamagnetic flow
terms 

The vorticity equation that is integrated in time is

.. math::

   \begin{aligned}\frac{\partial \Omega}{\partial t} =& \nabla\cdot\left(\mathbf{b}\sum_s Z_s n_sV_{||s}\right) \\
   &+ \underbrace{\nabla\cdot\left(\nabla\times\frac{\mathbf{b}}{B}\sum_s p_s\right)}_{\textrm{if diamagnetic}} + \underbrace{\nabla\cdot\mathbf{J_{exb}}}_{\mathrm{if exb\_advection}} \\
   &+ \nabla\cdot\left(\mathbf{b}J_{extra}\right)\end{aligned}

The nonlinearity :math:`\nabla\cdot\mathbf{J_{exb}}` is part of the
divergence of polarisation current. In its simplified form when
``exb_advection_simplified = true``, this is the :math:`E\times B`
advection of vorticity:

.. math::

   \nabla\cdot\mathbf{J_{exb}} = -\nabla\cdot\left(\Omega \mathbf{V}_{E\times B}\right)

When ``exb_advection_simplified = false`` then the more complete
(Boussinesq approximation) form is used:

.. math::

   \nabla\cdot\mathbf{J_{exb}} = -\nabla\cdot\left[\frac{\overline{A}}{2B^2}\nabla_\perp\left(\mathbf{V}_{E\times B}\cdot\nabla \hat{p}\right) + \frac{\Omega}{2} \mathbf{V}_{E\times B} + \frac{\overline{A}\overline{n}}{2B^2}\nabla_\perp^2\phi\left(\mathbf{V}_{E\times B} + \frac{\mathbf{b}}{B}\times\nabla\hat{p}\right) \right]
   
The form of the vorticity equation is based on `Simakov & Catto
<https://doi.org/10.1063/1.1623492>`_ (corrected in `erratum 2004
<https://doi.org/10.1063/1.1703527>`_), in the Boussinesq limit and
with the first term modified to conserve energy. In the limit of zero
ion pressure and constant :math:`B` it reduces to the simplified form.

.. doxygenstruct:: Vorticity
   :members:

relax_potential
~~~~~~~~~~~~~~~

This component evolves a vorticity equation, similar to the ``vorticity`` component.
Rather than inverting an elliptic equation at every timestep, this component evolves
the potential in time as a diffusion equation.

.. doxygenstruct:: RelaxPotential
   :members:

.. _electromagnetic:

electromagnetic
~~~~~~~~~~~~~~~

**Notes**: When using this module,

1. Set ``sound_speed:alfven_wave=true`` so that the shear Alfven wave
   speed is included in the calculation of the fastest parallel wave
   speed for numerical dissipation.
2. For tokamak simulations use Neumann boundary condition on the core
   and Dirichlet on SOL and PF boundaries by setting
   ``electromagnetic:apar_core_neumann=true`` (this is the default).
3. Set the potential core boundary to be constant in Y by setting
   ``vorticity:phi_core_averagey = true``
4. Magnetic flutter terms must be enabled to be active
   (``electromagnetic:magnetic_flutter=true``).  They use an
   ``Apar_flutter`` field, not the ``Apar`` field that is calculated
   from the induction terms.
5. If using ``vorticity:phi_boundary_relax`` to evolve the radial
   boundary of the electrostatic potential, the timescale
   ``phi_boundary_timescale`` should be set much longer than the
   Alfven wave period or unphysical instabilities may grow from the
   boundaries.

This component modifies the definition of momentum of all species, to
include the contribution from the electromagnetic potential
:math:`A_{||}`.

Assumes that "momentum" :math:`p_s` calculated for all species
:math:`s` is

.. math::

   p_s = m_s n_s v_{||s} + Z_s e n_s A_{||}

which arises once the electromagnetic contribution to the force on
each species is included in the momentum equation. This requires
an additional term in the momentum equation:

.. math::

   \frac{\partial p_s}{\partial t} = \cdots + Z_s e A_{||} \frac{\partial n_s}{\partial t}

This is implemented so that the density time-derivative is calculated using the lowest order
terms (parallel flow, ExB drift and a low density numerical diffusion term).

The above equations are normalised so that in dimensionless quantities:

.. math::

   p_s = A n v_{||} + Z n A_{||}

where :math:`A` and :math:`Z` are the atomic number and charge of the
species.

The current density :math:`j_{||}` in SI units is

.. math::

   j_{||} = -\frac{1}{\mu_0}\nabla_\perp^2 A_{||}

which when normalised in Bohm units becomes

.. math::

   j_{||} = - \frac{1}{\beta_{em}}\nabla_\perp^2 A_{||}

where :math:`\beta_{em}` is a normalisation parameter which is half
the plasma electron beta as normally defined:

.. math::

   \beta_{em} = \frac{\mu_0 e \overline{n} \overline{T}}{\overline{B}^2}

To convert the species momenta into a current, we take the sum of
:math:`p_s Z_s e / m_s`. In terms of normalised quantities this gives:

.. math::

   - \frac{1}{\beta_{em}} \nabla_\perp^2 A_{||} + \sum_s \frac{Z^2 n_s}{A}A_{||} = \sum_s \frac{Z}{A} p_s

The toroidal variation of density :math:`n_s` must be kept in this
equation.  By default the iterative "Naulin" solver is used to do
this: A fast FFT-based method is used in a fixed point iteration,
correcting for the density variation.

Magnetic flutter terms are disabled by default, and can be enabled by setting

.. code-block:: ini

   [electromagnetic]
   magnetic_flutter = true

This writes an ``Apar_flutter`` field to the state, which then enables perturbed
parallel derivative terms in the ``evolve_density``, ``evolve_pressure``, ``evolve_energy`` and
``evolve_momentum`` components. Parallel flow terms are modified, and parallel heat
conduction.

.. math::

   \begin{aligned}\mathbf{b}\cdot\nabla f =& \mathbf{b}_0\cdot\nabla f + \delta\mathbf{b}\cdot\nabla f \\
   =& \mathbf{b}_0\cdot\nabla f + \frac{1}{B}\nabla\times\left(\mathbf{b}A_{||}\right)\cdot\nabla f \\
   \simeq& \mathbf{b}_0\cdot\nabla f + \frac{1}{B_0}\left[A_{||}\nabla\times\mathbf{b} + \left(\nabla A_{||}\right)\times\mathbf{b}_0\right]\cdot\nabla f \\
   \simeq& \mathbf{b}_0\cdot\nabla f + \frac{1}{B_0}\mathbf{b}_0\times \nabla A_{||} \cdot \nabla f\end{aligned}

.. doxygenstruct:: Electromagnetic
   :members:
