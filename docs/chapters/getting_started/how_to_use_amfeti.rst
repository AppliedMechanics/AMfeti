=================
How To Use AMfeti
=================

In this chapter we explain how the use of AMfeti was
envisioned and how one could get the best out of the AMfeti framework.

Generally, AMfeti is used for highly complex problems, for research purposes.
The reason for this is that the efficiency increase provided by
parallelization is very valuable for such problems.

FETI Solvers
----------------

Possible FETI solvers include:

* LinearStaticFetiSolver.

  * Required parameters:

    * dictionary object containing stiffness-matrices

    * dictionary object containing connectivity matrices with corresponding interface IDs

    * dictionary object containing external forces

    * *optional*: arguments for solver-configuration


* NonlinearStaticFetiSolver

  * Required parameters:

    * dictionary object containing callbacks for stiffness-matrices

    * dictionary object containing connectivity matrices with corresponding interface IDs

    * dictionary object containing callbacks internal forces

    * dictionary object containing callbacks external forces

    * dictionary object containing start values for local solutions (e.g. displacements)

    * *optional*: arguments for solver-configuration


* LinearDynamicFetiSolver

  * Required parameters:

    * dictionary object containing IntegratorBase objects describing the dynamic behavior of the local problems. For details see :ref:`Basics of time-integration<time_integration.rst>`.

    * dictionary object containing connectivity matrices with corresponding interface IDs

    * start-time as type double

    * end-time as type double

    * dictionary object containing start values for local solutions (e.g. displacements)

    * dictionary object containing start values for local solutions' first time-derivative (e.g. velocity)

    * dictionary object containing start values for local solutions' second time-derivative (e.g. acceleration)

    * *optional*: arguments for solver-configuration


* NonlinearDynamicFetiSolver

  * Required parameters:

    * dictionary object containing IntegratorBase objects describing the dynamic behavior of the local problems. For details see :ref:`Basics of time-integration<time_integration.rst>`.

    * dictionary object containing connectivity matrices with corresponding interface IDs

    * start-time as type double

    * end-time as type double

    * dictionary object containing start values for local solutions (e.g. displacements)

    * dictionary object containing start values for local solutions' first time-derivative (e.g. velocity)

    * dictionary object containing start values for local solutions' second time-derivative (e.g. acceleration)

    * *optional*: arguments for solver-configuration

We provide detailed information on the syntax of FETI solvers and how to use them
with some :ref:`Examples<examples>`.

Importing the Mesh of Interest
-------------------------------

The creation of a Mesh is up to the users themselves.
They can create their .msh files as they wish
(e.g. with `gmsh <https://gmsh.info/>`_)
and then these files can be loaded for preprocessing with AMfe.


Set Up Component
----------------

The next step is setting up components or
mechanical systems from externally imported meshes. Furthermore,
the material as well as the boundary conditions can/should be specified as well.

Afterwards, information about the stiffness matrix, the connectivity matrix
and the external forces needs to be obtained, as this information will be necessary
for the FETI solvers. Dynamic FETI solvers also require
information about the time (start-time and end-time). Nonlinear FETI solvers
also require start-values for local solutions (e.g. displacement) and
sometimes their derivatives (e.g. velocity).

Further detailed explanation on each FETI solver, as well as their requirements
and usage can be found in the :ref:`Examples<examples>` and
:ref:`Fundamentals<fundamentals>`.

Solve Problem
-------------

Before solving the problem, the user needs to choose the method which will
be used for the particular problem. The user can choose between the
**Preconditioned Conjugare Projected Gradient method (PCPG solver)** and
the **Generalized Minimal RESidual method (GMRES solver)**. The default setting
for the FETI solvers is the PCPG method.

The FETI solver then uses the method as well as the information of
the substructured system (stiffness matrix, connectivity matix, external forces..).
We remind the users that the FETI solver is chosen in accordance with the problem at hand.


Visualization
--------------

After solving, we recommend that users unpack their solution (usually component-wise)
save it as separate files. The solution can then be visualized with the help
of a scientific visualization software like
`ParaView <https://www.paraview.org/Wiki/ParaView>`_.
For this purpose, AMfe provides exporter routines.
