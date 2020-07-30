AMfeti - FETI Research Code at the Chair of Applied Mechanics
=============================================================

(c) 2020 Lehrstuhl für Angewandte Mechanik, Technische Universität München

# AMfeti
AMfeti is a Python Library to solve and implement parallel FETI-Like solvers using mpi4py.


Overview:
---------

1.  [Installation](#installation-of-amfeti)
2.  [Documentation](#documentation)
3.  [Workflow](#workflow-for-pre--and-postprocessing)
4.  [Hints](#hints)


Installation of AMfeti
----------------------

Before installing AMfeti we stronly recommend the use of [ANACONDA](https://www.anaconda.com/distribution/) and 
[git](https://git-scm.com/downloads).
AMfeti is supposed to work in both Windows and Linux system, but is not fully supported, so please let us know if you 
are facing any problem.

# Dependecies
   - Python version 3.7 or higher
   - `numpy`, `scipy`, `mpi4pys`, `pandas`, `matplotlib`, `dill`
   - for building the documentation `sphinx`, `numpydoc`, `sphinx_rtd_theme`
   - for testing `nose`
   - for checking the code readability: `pylint`

# Installation
We recommend to create a separate environment in anaconda for your amfeti installation.
Then you later have the opportunity to create a new environment for other projects that can have different
requirements (such as python 2.7 instead 3.7). 

For getting the package type
    ```{r, engine='bash', count_lines}
    git clone https://gitlab.lrz.de/AM/AMfeti.git
    ```
in your console. Git will clone the repository into the current folder.
For installing the package in development mode run
    ```{r, engine='bash', count_lines}
    cd AMfeti
    conda create --name <environment-name-of-choice> python=3.7
    conda activate <environment-name-of-choice> 
    ```
In current Anaconda-versions under Linux the previous command might not work. You'll have to activate your environment 
via the following command instead
    ```{r, engine='bash', count_lines}
    source activate <environment-name-of-choice>
    ```
The following command then installs AMfeti into your environment.
    ```{r, engine='bash', count_lines}
    python setup.py develop
    ```
This way it is importable in any Python-script on your computer, if the associated environment is activated.

After the installation of AMfeti, you should run all unittests to make sure everything is properly working.
```{r, engine='bash', count_lines}
cd amfeti/tests
nosetests
```

We aim to cover all source files with unittests, so feel free to run all of them.
AMfeti uses mpi4pi and requires the installation of some mpi distriction, see 
[MSMPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi),
[IntelMPI](https://software.intel.com/en-us/mpi-library), and [OpenMPI](https://www.open-mpi.org/). Because multiple 
MPI implementation are supported, the user must create an environment variable to set MPI path that must be used in 
AMfeti.

```{r, engine='bash', count_lines}
export MPIDIR=/program/mpi
```

Also, you can have multiple Python virtual environments. In that case you must set an environment variable to specify 
which python.exe to use:

```{r, engine='bash', count_lines}
export 'PYTHON_ENV'=/condaenv/amfeti
```

Now, it is time to run python and import amfeti modules.

```{r, engine='bash', count_lines}
python
>>> import amfeti
```

Have fun!

Documentation
-------------
Further documentation to this code is in the folder `docs/`. For building the documentation, go to the `docs/` folder
and type

    make html

The documentation will be built in the folder `docs/` available as html in `_build`.
If the command above does not work, the execution of `python setup.py build_sphinx` in the main-folder
also builds the documentation.


Hints
-----

#### Python and the Scientific Ecosystem


Though Python is a general purpose programming language, it provides a great ecosystem for scientific computing.
As resources to learn both, Python as a language and the scientific Python ecosystem,
the following resources are recommended to become familiar with them.
As these topics are interesting for many people on the globe, lots of resources can be found in the internet.

##### Python language:
- [A byte of Python:](http://python.swaroopch.com/) A good introductory tutorial to Python. My personal favorite.
- [Learn Python the hard way:](http://learnpythonthehardway.org/book/) good introductory tutorial to the programming language.
- [Youtube: Testing in Python ](https://www.youtube.com/watch?v=FxSsnHeWQBY) This amazing talk explains the concept
and the philosophy of unittests, which are used in the `amfe` framework.

##### Scientific Python Stack (numpy, scipy, matplotlib):
- [Scipy Lecture Notes:](http://www.scipy-lectures.org/) Good and extensive lecture notes which are evolutionary improved online with very good reference on special topics, e.g. sparse matrices in `scipy`.
- [Youtube: Talk about the numpy data type ](https://www.youtube.com/watch?v=EEUXKG97YRw) This amazing talk **is a must-see** for using `numpy` arrays properly. It shows the concept of array manipulations, which are very effective and powerful and extensively used in `amfe`.
- [Youtube: Talk about color maps in matplotlib](https://youtu.be/xAoljeRJ3lU?list=PLYx7XA2nY5Gcpabmu61kKcToLz0FapmHu) This interesting talk is a little off-topic but cetainly worth to see. It is about choosing a good color-map for your diagrams.
- [Youtube: Talk about the HDF5 file format and the use of Python:](https://youtu.be/nddj5OA8LJo?list=PLYx7XA2nY5Gcpabmu61kKcToLz0FapmHu) Maybe of interest, if the HDF5 data structure, in which the simulation data are extracted, is of interest. This video is no must-have.

##### Version Control with git:
- [Cheat sheet with the important git commands](https://www.git-tower.com/blog/git-cheat-sheet/) Good cheatsheet with all the commands needed for git version control.
- [Youtube: git-Workshop](https://youtu.be/Qthor07loHM) This workshop is extensive and time intensive but definetely worth the time spent. It is a great workshop introducing the concepts of git in a well paced manner ([The slides are also available](https://speakerdeck.com/singingwolfboy/get-started-with-git)).
- [Youtube: git-Talk](https://youtu.be/ZDR433b0HJY) Very fast and informative technical talk on git. Though it is a little bit dated, it is definitely worth watching. 

#### IDEs:

A good IDE to start with is Spyder, which has sort of a MATLAB-Style look and feel.
It is part of anaconda ans provides nice features like built-in debugging, static code analysis with pylint and a
profiling tool to measure the performance of the code.

Other editors integrate very well into Python like Atom.

I personally work with PyCharm, which is an IDE for Python. However as it provides many functions one could be
overwhelmed by it at first. 

---------------------------------------
**Hint** 

On Mac OS X `Spyder 2` may run very slow, as there are some issues with the graphical frontent library, pyqt4. These issues are resolved on `Spyder 3` by using pyqt5, which can already be installed on anaconda as beta version resolving all these issues. To install `Spyder 3`, use either 

    conda update qt pyqt
    conda install -c qttesting qt pyqt
    conda install -c spyder-ide spyder==3.0.0b6
   
or (which worked better for me)
   
    pip install --pre -U spyder

-------------------------------------

#### Profiling the code

a good profiling tool is the cProfile module. It runs with

    python -m cProfile -o stats.dat myscript.py

The stats.dat file can be analyzed using the `snakeviz`-tool which is a Python tool which is available via `conda` or `pip` and runs with a web-based interface. To start run

    snakeviz stats.dat

in your console.


Theory behind AMfeti
--------------------
## Solving with Dual Assembly
The AMfeti library is intend to provide easy functions in order to solve, the dual assembly problem, namely:


$$
\begin{bmatrix} K & B^{T} \\
                 B & 0  
\end{bmatrix}
\begin{bmatrix} q \\ 
\lambda \end{bmatrix}
=
\begin{bmatrix} f \\ 
0 \end{bmatrix}
$$

Generally the block matrix $K$ is singular due to local rigid body modes, then the inner problem is regularized by 
adding a subset of the inter-subdomain compatibility requirements:


$$
\begin{bmatrix} K & B^TG^{T} & B^{T} \\
                GB & 0 & 0   \\
                B & 0 & 0   \\
\end{bmatrix}
\begin{bmatrix} q \\ 
\alpha \\
\lambda \end{bmatrix}
=
\begin{bmatrix} f \\ 
0 \\
0 \end{bmatrix}
$$

Where $G$ is defined as $-R^TB^T$.

The Dual Assembly system of equation described above can be separated into two equations.

\begin{equation}
Kq + B^{T}\lambda  = f \\
Bu = 0 
\end{equation}

Then, the solution u can be calculated by:

\begin{equation}
u =  K^*(B^{T}\lambda  + f) +  R\alpha \\
\end{equation}

Where $K^*$ is the generelized pseudo inverse and $R$ is $Null(K) = \{r \in R: Kr=0\}$, named the kernel of the K matrix.
In order to solve for $u$ the summation of all forces in the subdomain, interface, internal and extenal forces must be 
in the image of K. This implies the $(B^{T}\lambda  + f)$ must be orthonal to the null space of K.

\begin{equation}
R(B^{T}\lambda  + f) = 0 \\
\end{equation}

Phisically, the equation above enforces the self-equilibrium for each sub-domain. Using the compatibility equation and 
the self-equilibrium equation, we can write the dual interface equilibrium equation as:


$$
\begin{bmatrix} F & G^{T} \\
                 G & 0  
\end{bmatrix}
\begin{bmatrix} \lambda  \\ 
\alpha
\end{bmatrix}
=
\begin{bmatrix} d \\ 
e \end{bmatrix}
$$

Where $F = BK^*B^T$, $G = -R^TB^T$, $d = BK^*f$ and $e =- R^Tf $.

## Further references
[1]  C. Farhat and F.-X. Roux (1991): A method of Finite Element Tearing and Interconnecting and its parallel solution 
        algorithm. International Journal for Numerical Methods in Engineering 31 1205--1227.

[2]  C. Farhat and D.J. Rixen (1999): A simple and efficient extension of a class of substructure based
        preconditioners to heterogeneous structural mechanics problems. International Journal for Numerical Methods in
        Engineering 44 489--516.
