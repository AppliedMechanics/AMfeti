"""
We are going to solve a 3D linear static system using the FETI
linear static solver in this example. We decided to use a 3D truss
with a slope on the south edge, which is going to be fixed on
the west edge and a force of 1.5 kN is going to be applied on
the north surface.

We start by importing the libraries needed for this example.
"""

import numpy as np
import logging

###########################################

from amfe.io.mesh import AmfeMeshConverter, GmshAsciiMeshReader
from amfe.material import KirchhoffMaterial
from amfe.component import StructuralComponent
from amfe.component.component_composite import MeshComponentComposite
from amfe.component.tree_manager import TreeBuilder
from amfe.neumann import FixedDirectionNeumann
from amfe.solver.translators import MulticomponentMechanicalSystem
from amfe import ui
from amfe.solver import AmfeSolution
from amfe.forces import constant_force

############################################

from amfeti import LinearStaticFetiSolver
from amfeti.solvers import PCPGsolver

"""
The mesh needs to be prepared in Python before it is passed to the
AMfeti solver.
"""

# Material Properties
E_alu = 70e3
nu_alu = 0.34
rho_alu = 2.7e0
my_material = KirchhoffMaterial(E_alu, nu_alu, rho_alu, thickness=1)

# we specify the path for the input mesh and an output path,
# where we will save the solutions
input_file = '/meshes/3d_truss_with_slope.msh'
output_path = '/results/linear_static_3d_example'

"""
We are setting up the material properties and defining our component 
with the mesh file. It's important to remember to set the parameter 
of ``surface2partition`` to ``True`` when reading the mesh.
"""

# Reading the .msh file and defining a structural component
reader = GmshAsciiMeshReader(input_file)
converter = AmfeMeshConverter()
reader.parse(converter, surface2partition=True)
my_mesh = converter.return_mesh()
my_component = StructuralComponent(my_mesh)

"""
We proceed by assigning the material properties and
mapping the global degrees of freedom for the Dirichlet boundary conditions.
"""

# Assigning material properties on physical group called "material"
ui.assign_material_by_group(my_component, my_material, 'material')

# Mapping the degrees of freedom for nodes belonging to the physical group called "dirichlet"
glo_dofs_x = my_component.mapping.get_dofs_by_nodeids(my_component.mesh.get_nodeids_by_groups(['dirichlet']), 'ux')
glo_dofs_y = my_component.mapping.get_dofs_by_nodeids(my_component.mesh.get_nodeids_by_groups(['dirichlet']), 'uy')
glo_dofs_z = my_component.mapping.get_dofs_by_nodeids(my_component.mesh.get_nodeids_by_groups(['dirichlet']), 'uz')

my_composite = MeshComponentComposite(my_component)

"""
We define a structural composite object with the help of the tree builder
that manages the substructures and the connections between them.
"""

# Decomposition of component
tree_builder = TreeBuilder()
tree_builder.add([0], [my_composite])
leaf_id = tree_builder.leaf_paths.max_leaf_id
tree_builder.separate_partitioned_component_by_leafid(leaf_id)
structural_composite = tree_builder.root_composite.components[0]
structural_composite.update_component_connections()

"""
Then we define the external force of 1.5 kN and apply the Neumann boundary condition.
"""

F = constant_force(1.5E3)

# Neumann conditions, with the force direction [0, -1, 0] for the [x direction, y direction, z direction]
my_neumann = FixedDirectionNeumann(np.array([0, -1, 0]), time_func=F)
structural_composite.assign_neumann('Neumann0', my_neumann, ['neumann'], '_groups')

# Dirichlet conditions
dirichlet = structural_composite.components[1]._constraints.create_dirichlet_constraint()
for dof in glo_dofs_x.reshape(-1):
    structural_composite.assign_constraint('Dirichlet0', dirichlet, np.array([dof], dtype=int), [])
for dof in glo_dofs_y.reshape(-1):
    structural_composite.assign_constraint('Dirichlet1', dirichlet, np.array([dof], dtype=int), [])
for dof in glo_dofs_z.reshape(-1):
    structural_composite.assign_constraint('Dirichlet2', dirichlet, np.array([dof], dtype=int), [])

"""
Now that we have finalized the structural composite, we can create a 
multicomponent mechanical system, i.e. a system consisting of substructures.
"""

# FETI-solver
substructured_system = MulticomponentMechanicalSystem(structural_composite, constant_mass=True, constant_damping=True, all_linear=True, constraint_formulation='boolean')

"""
Since this is a linear static problem, we'd like to use the LinearStaticFetiSolver.
However, this solver requires dictionaries for the K matrices, 
the B matrices and the f_ext. For this purpose, we write a wrapper function 
that prepares these dictionaries, we need to pass to the FETI solver.
"""

def _create_K_B_f_dict(B_dict, msys_dict):
    # Initialize dictionaries
    K_dict_trans = dict()
    B_dict_trans = dict()
    f_ext_dict_trans = dict()
    interface_dict = dict()
    int_num = 1
    system_wrapper = dict()

    # Define a wrapper class
    class SystemWrapper:
        def __init__(self, msystem):
            self.msystem = msystem
            self.f_ext_stored = None

        def K(self, q):
            return self.msystem.K(q, np.zeros_like(q), 0)

        def f_ext(self, q):
            if self.f_ext_stored is None:
                self.f_ext_stored = self.msystem.f_ext(q, np.zeros_like(q), 0)
            return self.f_ext_stored

    for i_system, msys in msys_dict.items():
        subs_key = i_system

        system_wrapper[subs_key] = SystemWrapper(msys)

        K_dict_trans[subs_key] = system_wrapper[subs_key].K(np.zeros(msys.dimension))
        f_ext_dict_trans[subs_key] = system_wrapper[subs_key].f_ext(np.zeros(msys.dimension))

        B_local = dict()
        for key in B_dict.keys():
            if key not in interface_dict:
                interface_dict[key] = 'interface' + str(int_num)
                interface_dict[(key[1], key[0])] = 'interface' + str(int_num)
                int_num += 1
            if key[0] == i_system:
                B_local[interface_dict[key]] = B_dict[key]
        B_dict_trans[subs_key] = B_local
    return K_dict_trans, B_dict_trans, f_ext_dict_trans

"""
We can now use this function to define the dictionaries for K, B 
and f_ext and call the linear static FETI solver.
"""

K_dict, B_dict, f_ext_dict = _create_K_B_f_dict(substructured_system.connections, substructured_system.mechanical_systems)

# Defining an instance of the Preconditioned Conjugate Projected Gradient (PCPG) solver as a global system solver to be used
global_solver = PCPGsolver()
global_solver.set_config({'full_reorthogonalization': True})
feti_solver = LinearStaticFetiSolver(K_dict, B_dict, f_ext_dict, global_solver=global_solver)
feti_solver.update()

"""
A solution object, containing all global solutions, solver-information 
and local problems, is returned by the solver.
"""

# Solving our system
solution_obj = feti_solver.solve()

"""
We now have our solution, but it's a solution object so we need to 
read it out and store the solution in a way that is readable to us. 
We are going to create ``.hdf5`` and ``.xdmf`` files that contain 
the results.
"""

# Initializing an empty dictionary
solution_writer = dict()

# Save all items from the solution object in the dictionary
for i_prob, local_problem in solution_obj.local_problems.items():
    solution_writer[i_prob] = AmfeSolution()
    q = local_problem.q
    msys = substructured_system.mechanical_systems[i_prob]
    if i_prob in substructured_system.constraint_formulations:
        formulation = substructured_system.constraint_formulations[i_prob]
        u, du, ddu = formulation.recover(q, np.zeros_like(q), np.zeros_like(q), 0)
    else:
        u = q
        du = np.zeros_like(u)
    strains, stresses = structural_composite.components[i_prob].strains_and_stresses(u, du, 0)
    solution_writer[i_prob].write_timestep(0, u, None, None, strains, stresses)

# Export the items in files readable in Paraview
for i_comp, comp in structural_composite.components.items():
    path = output_path + '/Component_' + str(i_comp)
    ui.write_results_to_paraview(solution_writer[i_comp], comp, path)