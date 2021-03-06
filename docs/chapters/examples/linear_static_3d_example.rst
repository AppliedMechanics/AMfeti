=========================
Linear Static 3D Example
=========================

We are going to solve a 3D linear static system using the FETI linear static solver in this example. We decided to
use a 3D truss with a slope on the south edge, which is going to be fixed on the west edge and a force of 1.5 kN is going to be
applied on the north surface. This structure will be meshed and partitioned into four substructures using gmsh (version: 3.0.6).

Note that the Python script for this example is available in ``docs/chapters/examples/linear_static_3d_example.py``.
Remember that you need to have AMfeti and AMfe installed in order to use the functions we use here. Also remember
to set the path for the input meshes and the path where the results will be saved to paths to folders on your computer
if you want to run the scripts we provide.


Preprocessing
===============

We have three physical groups:

* "dirichlet" - for a Dirichlet boundary on the north edge,

* "neumann" - for a Neumann boundary on the south edge,

* "material" - for the material on the whole 3D volume.

Finally, the mesh was partitioned into four substructures, which are shown with different colors in the image below.
|

.. image:: images/linear_static_3d_example_gmsh_conditions.png
    :width: 800
    :align: center

|

Now that we have our mesh, we need to prepare it using Python before using the AMfeti solver. We start by importing
the libraries needed for this example.


Afterwards, we are setting up the material properties and defining our component with the mesh file.
It's important to remember to set the parameter of ``surface2partition`` to ``True`` when reading the mesh.


We proceed by assigning the material properties and
mapping the global degrees of freedom for the Dirichlet boundary conditions.


We define a structural composite object with the help of the tree builder
that manages the substructures and the connections between them.


Then we define the external force of 1.5 kN and apply the Neumann boundary condition.


FETI Solver
=============

Now that we have finalized the structural composite, we can create a multicomponent mechanical system, i.e. a system
consisting of substructures.


Since this is a linear static problem, we'd like to use the LinearStaticFetiSolver.
However, this solver requires dictionaries for the K matrices, the B matrices and the f_ext. For this purpose, we
write a wrapper function that prepares these dictionaries, we need to pass to the FETI solver.


We can now use this function to define the dictionaries for K, B and f_ext and call the linear static FETI solver.


A solution object, containing all global solutions, solver-information and local problems, is returned by the solver.

Postprocessing
==============

We now have our solution, but it's a solution object so we need to read it out and store the solution in a
way that is readable to us. We are going to create ``.hdf5`` and ``.xdmf`` files that contain the results.


Finally, we can take a look at the solution. For this, we use Paraview v5.7.0. The original 3D object can be seen on the left
and the deformed 3D object can be seen on the right.

|

.. image:: images/linear_static_3d_example_results.png
    :width: 800
    :align: center

|