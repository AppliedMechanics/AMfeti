.. AMfeti documentation master file, created by
   sphinx-quickstart on Tue Jun  9 20:54:07 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

####################
AMfeti Documentation
####################


.. rubric:: The AMfeti Documentation

AMfeti is built in order to provide a simple and flexible library for parallelization with domain decomposition methods,
such as FETI.

Release notes
-------------

API changes and new features can be found in the :doc:`Release Notes<release/index>`

**********************************
How the documentation is organized
**********************************

    * :doc:`Tutorials (Getting started Guide)<chapters/getting_started/index>`
      are a good starting point to learn how to use AMfeti. Start here if you are
      new to AMfeti.

    * :doc:`Fundamentals (Topic guides)<chapters/fundamentals/index>`
      explains different parts and concepts of AMfeti. It is the heart of
      documentation and mostly intended for users that are familiar with basics
      of AMfeti or users that have already done the Tutorials

    * :doc:`Examples<chapters/examples/index>` shows some examples for
      different tasks. This part of documentation can be used if is interested
      in how to solve specific problem. For many problems an example can be
      found here that can be altered for own needs.

    * :doc:`Reference<chapters/package_doc/index>` is the API documentation
      of the whole package.




The idea
--------

The AMfeti package is mainly understood as a library for solvers based on domain-decomposition-methods and linear
algebra. It is not a standalone Finite Elements program. However, this enables the user to freely choose pre- and
postprocessing tools and use AMfeti as solver. AMfeti is mainly based on the Python programming language. This makes it
very flexible and easy to maintain, which is essential in research. The parallel computing is done via MPI
(Message Passing Interface). Hence it is not the most performant solver you can get, but still provides decent
calculation times and makes a good compromise between maintainability, usability and performance.
