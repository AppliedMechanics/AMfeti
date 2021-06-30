==========================
Basics of time-integration
==========================


Structural dynamics problems
============================

The system-behavior is usually expressed by an inhomogeneous differential-equation in time up to second
order.

.. math::

    \textbf{M}\frac{d\vec{q}}{dt^2}+\textbf{C}\frac{d\vec{q}\ }{dt}+\textbf{K}\vec{q}-\vec{f}_{ext}(t)=\vec{0}


The solution of this system is then achieved in a time-stepping-manner where the solution at a following time-step is
received only from information of the previous time-step.

by a linearized system-matrix a.k.a. Jacobian-matrix and two separated parts of the system's residual-
function. Moreover the time-integration has to be performed in a prediction-correction manner.