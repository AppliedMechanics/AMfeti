.. _pre_installation_steps.rst:

========================
Pre- Installation Guide
========================

Install Python
--------------

.. image:: images/python-logo.png
    :width: 600

| AMfeti requires a Python distribution 3.7 or newer. If you already have Python older than 3.7 installed,
make sure to also install a newer version and build the virtual environment with your newer Python version.
For the management of multiple independent Python-installations we recommend using Anaconda,
which is a package management and deployment framework that also includes a distribution of Python.
So, if you are also planning to install Anaconda,then you can directly install Anaconda
because its installation contains Python.


Unix Distributions
^^^^^^^^^^^^^^^^^^^

Most Linux distributions (e.g. Ubuntu, Debian..) have the newest Python distribution installed by default.
Linux users can check if they already have a Python distribution installed by opening the terminal
(with ``Ctrl+Alt+T``) and executing the command ``python``.


You can visit the `Python Unix page <https://docs.python.org/3/using/unix.html>`_ for tips on
installing the latest version on Unix platforms.


Windows Distributions
^^^^^^^^^^^^^^^^^^^^^

For Windows, the Python installers can be found on the `Python Windows page <https://www.python.org/downloads/windows/>`_.
We recommend to find the newest stable executable installer, download it and then double click on the
downloaded file to run the installation. Again, make sure that you are installing a version **newer than 3.7**.
Follow the instructions in the pop-up window until you finish the installation. Further details and customizable
installations can be found on the `Python Windows Launcher page <https://docs.python.org/3/using/windows.html#launcher>`_.

Mac Distributions
^^^^^^^^^^^^^^^^^

*Note for Mac users:*
*The parallel functionality of AMfeti, which uses an MPI-installation (e.g. OpenMPI), has not
been tested on Mac. So if you want to use AMfeti on a device with a Mac OS, you will have to
either only use the serial solvers or ensure that mpi4py works on your machine.*

For Mac, the Python installers can be found on the `Python Mac page <http://www.python.org/downloads/mac-osx/>`_.
We recommend to find the newest stable installer, download it and perform the installation.
Again, make sure that you are installing a version **newer than 3.7**.

Install Python IDE
-------------------

The most convenient way to write code is by using an IDE (Integrated Development Environment).
There are a big number of IDE options. Some well known IDEs include: PyCharm, Visual Studio Code, Spyder, IDLE, PyDev.

We recommend PyCharm for everyone interested in development. The Community version of PyCharm is available for free
and the Professional version of PyCharm available for free for educational purposes.
An installation guide can be found on the `JetBrains page <http://www.jetbrains.com/help/pycharm/installation-guide.html>`_.

Install Anaconda
-----------------

We highly recommend using Anaconda as a management tool for Python
virtual environments and loading Python packages. The most convenient
way of installing AMfeti into your Python-environment is using the
automatic installation scripts of AMfeti. Those scripts expect an
anaconda-environment to load all necessary libraries and their correct
versions, though. If you feel comfortable with other package management
tools, like ``pip`` — feel free to use them instead of Anaconda, but you
might need to install necessary packages manually.

Creating virtual environments and managing Python packages (e.g. Numpy, Scipy ..)
is done easily with an Anaconda distribution.
Anaconda is available for Windows, Linux and MacOS. An important requirement for the
installation is that at least 5GB of disk space are available.
The installation details can be found on the
`Anaconda Installation page <http://docs.anaconda.com/anaconda/install/>`_.

We recommend to add Anaconda to your PATH-variable.
