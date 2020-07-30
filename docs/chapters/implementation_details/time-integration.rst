Requirements to an Integrator-Class
===================================


Need for integrator-class
-------------------------
If a dynamic problem needs to be solved, AMfeti provides preconfigured dynamic FETI-solvers with dynamic local problems.
These :py:mod:`dynamic local problems <amfeti.local_problems.dynamic_local_problems>` require an integrator-object that
describes the local problem's dynamics.

Expected API and behavior of integrator-class
---------------------------------------------
This integrator-object needs the same API as the provided template
:py:class:`IntegratorBase <amfeti.local_problems.integrator_base.IntegratorBase>`. The integrator-classes of the
`AMfe project <https://github.com/AppliedMechanics/AMfe>` from version 1.1.0 on also exhibit this API and can be
used for the construction of local problems.
