Requirements of the integrator-class
=====================================

The FETI solvers used for dynamic problems require a description of the local dynamics of the
problem being solved. So if the user has a dynamic problem, they would need to provide
integrator-objects as inputs of the LinearDynamicFetiSolver or NonlinearDynamicFetiSolver.
In order to simplify the use of these FETI solvers, in this section we want to provide information
on the integrator-objects that are required by the solvers.

Integrator Base Class
-----------------------

The AMfeti library provides an Integrator-Base class,
which is a template for creating the integrator-objects. This base class has attributes
and methods that correspond to the ones expected by the local dynamic solvers.

The attributes in this class are:

* :py:mod:`dt` - the time step size [float]

* :py:mod:`_t_n` - previous time [float]

* :py:mod:`_q_n` - primary solution of previous time step [ndarray]

* :py:mod:`_dq_n` - first time-derivative of primary solution of previous time step [ndarray]

* :py:mod:`_ddq_n` - second time-derivative of primary solution of previous time step [ndarray]

* :py:mod:`t_p` - next (predicted) time [float]

* :py:mod:`q_p` - primary solution of next time step (predicted) [ndarray]

* :py:mod:`dq_p` - first time-derivative of primary solution of next time step (predicted) [ndarray]

* :py:mod:`ddq_p` - second time-derivative of primary solution of next time step (predicted) [ndarray]

The methods in this class are:

* :py:mod:`residual_int(dq_p)` - computes internal component of the residual, requires the velocity of the next time step as an input

* :py:mod:`residual_ext(dq_p)` - computes external component of the residual, requires the velocity of the next time step as an input

* :py:mod:`jacobian(dq_p)` - computes Jacobian matrix of the velocity, requires the velocity of the next time step as an input

* :py:mod:`set_prediction(q_n, dq_n, ddq_n, t_n)` - computes prediction of solution and its first and second derivatives, requires only information of the previous time-step and from that it generates estimates of the next solutions

* :py:mod:`set_correction(dq_p)` - corrects prediction of solution and its first and second derivatives, requires the updated velocity of the next time step as an input

The important thing to keep in mind when implementing an integrator class is that the integrator class
needs to be velocity based, meaning that the jacobian needs to be a derivative for velocities and
the methods' arguments are velocities. The user is free to implement any integrator scheme as long as
it is a velocity based one and has the method signatures and parameters we mentioned above.
The `AMfe library <https://github.com/AppliedMechanics/AMfe>`_ provides classes for a
velocity generalized alpha integrator, velocity generalized beta integrator
and other integrators. These classes have an API suitable for the AMfeti solvers.

The reason the residual is split into an internal and external part is that, in the
nonlinear case, the load stepping doesn't have to be handled by the integrator class but
can be handled in the local problem. A prefactor is then applied to the external residual,
but not the internal one.

