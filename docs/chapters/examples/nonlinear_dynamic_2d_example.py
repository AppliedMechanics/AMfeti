"""
We are going to solve a nonlinear dynamic system using the FETI
nonlinear dynamic solver in this example. We decided to use an
aluminium rectangular 2D structure, which is going to be fixed
on the north edge and a triangular force with a peak of 50kN
for 0.003sec.

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
from amfe.forces import triangular_force
from amfe.solver.integrator import VelocityGeneralizedAlpha
############################################

from amfeti import NonlinearDynamicFetiSolver
from amfeti.solvers import PCPGsolver

############################################
logging.basicConfig(level=logging.INFO)

"""
The mesh needs to be prepared in Python before it is passed to the
AMfeti solver.
"""

E_alu = 70e3
nu_alu = 0.34
rho_alu = 2.7e0

# we specify the path for the input mesh and an output path,
# where we will save the solutions
input_file = '/meshes/3d_truss_with_slope.msh'
output_path = '/results/nonlinear_dynamic_2d_example'

"""
We are setting up the material properties and defining our 
component with the mesh file. It's important to remember to 
set the parameter of ``surface2partition`` to ``True`` when 
reading the mesh.
"""

# Reading the .msh file and defining a structural component
reader = GmshAsciiMeshReader(input_file)
converter = AmfeMeshConverter()
reader.parse(converter, surface2partition=True)
my_mesh = converter.return_mesh()
my_component = StructuralComponent(my_mesh)

my_material = KirchhoffMaterial(E_alu, nu_alu, rho_alu, thickness=1)

"""
We proceed by assigning the material properties and mapping the global 
degrees of freedom for all nodes that lie on the Dirichlet boundary.
This mapping will be used to assign the Dirichlet constraints later.
"""

# Assigning material properties on physical group called "material"
ui.assign_material_by_group(my_component, my_material, 'material')

# Mapping the degrees of freedom for nodes belonging to the physical group called "dirichlet"
glo_dofs_x = my_component.mapping.get_dofs_by_nodeids(my_component.mesh.get_nodeids_by_groups(['dirichlet']), 'ux')
glo_dofs_y = my_component.mapping.get_dofs_by_nodeids(my_component.mesh.get_nodeids_by_groups(['dirichlet']), 'uy')

my_composite = MeshComponentComposite(my_component)

"""
We define a structural composite object with the help of the tree 
builder that manages the substructures and the connections between them.
"""

tree_builder = TreeBuilder()
tree_builder.add([0], [my_composite])
leaf_id = tree_builder.leaf_paths.max_leaf_id
tree_builder.separate_partitioned_component_by_leafid(leaf_id)

structural_composite = tree_builder.root_composite.components[0]
structural_composite.update_component_connections()

"""
Then we define a triangular force that is applied from
time=0.0s until time=0.002s with a peak of 50kN at time=0.001s 
and apply the Neumann boundary condition on the physical group 
of choice. Finally, we apply the Dirichlet conditions as well.
"""

# Neumann conditions, with the force direction [0, -1, 0] for the [x direction, y direction, z direction]
F = triangular_force(0.0, 0.001, 0.002, 5.0e4)
my_neumann = FixedDirectionNeumann(np.array([0, -1]), time_func=F)
structural_composite.assign_neumann('Neumann0', my_neumann, ['neumann'], '_groups')

# Dirichlet conditions
dirichlet = structural_composite.components[1]._constraints.create_dirichlet_constraint()
for dof in glo_dofs_x.reshape(-1):
    structural_composite.assign_constraint('Dirichlet0', dirichlet, np.array([dof], dtype=int), [])
for dof in glo_dofs_y.reshape(-1):
    structural_composite.assign_constraint('Dirichlet1', dirichlet, np.array([dof], dtype=int), [])

"""
Now that we have finalized the structural composite, we can create 
a multicomponent mechanical system, i.e. a system consisting of 
substructures. Have in mind that the ``all_linear`` parameter 
needs to be set to ``False`` here, because the problem is not linear.
"""

# FETI-solver
substructured_system = MulticomponentMechanicalSystem(structural_composite, constant_mass=True, constant_damping=True,
                                                      all_linear=False, constraint_formulation='boolean')

"""
Since this is a nonlinear dynamic problem, we'd like to use the 
NonlinearDynamicFetiSolver. However, this solver requires a dictionary 
of integrator objects, which are not directly provided by the 
MulticomponentMechanicalSystem and needs to be created and wrapped 
into that dictionary first. Such an integrator object describes the 
local dynamic behaviour. Moreover a dictionary containing the B matrices 
and starting values for the local solutions as well as their first and 
second time derivatives. For this purpose, we need to write a function 
that prepares these dictionaries, we need to pass to the FETI solver.
"""

def _create_integrator_B_dict(B_dict, msys_dict):
    integrator_dict = dict()
    B_dict_trans = dict()
    interface_dict = dict()
    q_0_dict = dict()
    dq_0_dict = dict()
    ddq_0_dict = dict()
    int_num = 1

    for i_system, msys in msys_dict.items():
        subs_key = i_system

        integrator_dict[subs_key] = VelocityGeneralizedAlpha(msys.M, msys.f_int, msys.f_ext, msys.K, msys.D)
        integrator_dict[subs_key].dt = 0.0001

        q_0_dict[subs_key] = np.zeros(msys.dimension)
        dq_0_dict[subs_key] = np.zeros(msys.dimension)
        ddq_0_dict[subs_key] = np.zeros(msys.dimension)

        B_local = dict()
        for key in B_dict.keys():
            if key not in interface_dict:
                interface_dict[key] = 'interface' + str(int_num)
                interface_dict[(key[1], key[0])] = 'interface' + str(int_num)
                int_num += 1
            if key[0] == i_system:
                B_local[interface_dict[key]] = B_dict[key]
        B_dict_trans[subs_key] = B_local
    return integrator_dict, B_dict_trans, q_0_dict, dq_0_dict, ddq_0_dict

"""
We can now use this function to define the dictionaries for the 
integrator objects, B matrices, the local solutions and their first 
and second derivatives and call the nonlinear dynamic FETI solver.
"""

integrator_dict, B_dict, q_0_dict, dq_0_dict, ddq_0_dict = _create_integrator_B_dict(
    substructured_system.connections, substructured_system.mechanical_systems)

logging.basicConfig(level=logging.INFO)

# Defining an instance of the Preconditioned Conjugate Projected Gradient (PCPG) solver as a global system solver to be used
solver = PCPGsolver()
solver.set_config({'full_reorthogonalization': True,
                   'save_history': True})
fetisolver = NonlinearDynamicFetiSolver(integrator_dict, B_dict, 0.0, 0.003, q_0_dict, dq_0_dict, ddq_0_dict, global_solver=solver,
                                                    loadpath_controller_options={'N_steps': 1,
                                                                              'nonlinear_solver_options': {'atol': 1.0e-06,
                                                                                                            'rtol': 1.0e-7,
                                                                                                            'max_iter': 10,
                                                                                                            'log_iterations': True}})
fetisolver.update()

"""
A solution object, containing all global solutions, solver-information 
and local problems, is returned by the solver.
"""

# Solving our system
solution_obj = fetisolver.solve()

"""
We now have our solution, but it's a solution object so we need to 
read it out and store the solution in a way that is readable to us. 
We are going to create ``.hdf5`` and ``.xdmf`` files that contain the results.
"""

# Initializing an empty dictionary
solution_writer = dict()

# Save all items from the solution object in the dictionary
for i_prob, local_problem in solution_obj.local_problems.items():
    solution_writer[i_prob] = AmfeSolution()
    t = local_problem.t
    q = local_problem.q
    dq = local_problem.dq
    ddq = local_problem.ddq
    msys = substructured_system.mechanical_systems[i_prob]
    for ts in range(len(t)):
        if i_prob in substructured_system.constraint_formulations:
            formulation = substructured_system.constraint_formulations[i_prob]
            u, du, ddu = formulation.recover(q[ts], dq[ts], ddq[ts], t[ts])
        else:
            u = q[ts]
            du = dq[ts]
            ddu = ddq[ts]
        strains, stresses = structural_composite.components[i_prob].strains_and_stresses(u, du, t[ts])
        solution_writer[i_prob].write_timestep(t[ts], u, du, ddu, strains, stresses)

# Export the items in files readable in Paraview
for i_comp, comp in structural_composite.components.items():
    path = '/Component_' + str(i_comp)
    ui.write_results_to_paraview(solution_writer[i_comp], comp, path)
