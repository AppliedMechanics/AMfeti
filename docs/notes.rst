*****
Notes
*****

How to run the AMfeti Code
==========================

You need the following stuff installed:

- `Python 3.5 <http://www.python.org>`_ or higher
- `numpy, scipy <http://www.scipy.org>`_
- `mpi4py`_
- Some Python-Packages in order to build this documentation
   - `sphinx <http://www.sphinx-doc.org/>`_
   - `numpydoc <https://pypi.python.org/pypi/numpydoc>`_

For Python exist several ways how to install it on your computer. We recommend to install Anaconda, which is a Package
manager and lets you install easily all additional packages and tools related to Python.

After installing the `Anaconda Manager <https://store.continuum.io/cshop/anaconda/>`_ (make sure, that you install the
Python 3 version of it) run in your bash-console

>>> conda install numpy
>>> conda install scipy
>>> conda install sphinx
>>> conda install numpydoc

For a Matlab-like Development center we recommend `Spyder <http://spyder-ide.blogspot.de>`_. Spyder can also easily be
installed via Anaconda:

>>> conda install spyder

For installing the code type

>>> python setup.py develop

to install the package with the ability to do changes in the source code right away. If you do not want to do any
developments, use

>>> python setup.py install


Getting into the code
"""""""""""""""""""""
For getting started and familiar with the code, we recommend to start with the examples. They show some cases that are
working and are not too complicated.


General Notes on the amfeti code
================================
The amfeti library has the goal to provide a fast to develop and simple domain decomposition-solver library for use in
research. Therefore the focus is on flexibility to adapt the code to new problems and to easily implement a new methods
and not so much on runtime performance.


Tips & Tricks:
==============

How to plot matrices in matplotlib:

>>> import matplotlib as mpl
>>> from matplotlib import pyplot as plt; import scipy as sp
>>> A = sp.random.rand(10, 10)
>>> plt.matshow(A)
>>> # In order to plot in log scale:
>>> plt.matshow(A, norm=mpl.colors.LogNorm())
>>> plt.colorbar()
>>> plt.set_cmap('jet') # 'jet' is default; others looking good are 'hot'

How to show the sparsity pattern of a sparse matrix :code:`A_csr`:

>>> plt.spy(A_csr, marker=',')

You can use different markers, as :code:`','` are pixels and very small, they make sense when large matrices are
involved. However, for small matrices, :code:`'.'` gives a good picture.

Plot on log scales:

>>> from matplotlib import pyplot as plt; import scipy as sp
>>> x = np.arange(200)
>>> # plot y with logscale
>>> plt.semilogy(x)
>>> # plot x with logscale
>>> plt.semilogx(x)
>>> # plot x and y in logscale
>>> plt.loglog(x)

Check out more on http://matplotlib.org/examples/color/colormaps_reference.html
