Installation Instructions
=========================

Pyphasefield is a pure-python package and thus may be installed through `PyPI`_ using the command ``pip install pyphasefield``. 
However, it is **strongly** recommended to install pyphasefield into an Anaconda_ installation, as this permits the easy use of two powerful 
packages which greatly increase the scope of what pyphasefield can do: `pycalphad`_ (Access arbitrary thermodynamics using Thermodynamic 
DataBase (TDB) files), and `numba`_ (Run accelerated phase field simulations using General Purpose GPU computing). 

Anaconda Installation Instructions
----------------------------------
This tutorial assumes you have installed Anaconda_ and (optionally) created and activated an `Anaconda environment`_ to contain these 
packages, separate from the base installation. After opening the Anaconda terminal ("Anaconda Prompt" on Windows) and activating your 
environment if necessary, run the following commands to install pyphasefield, as well as recommended (but not required) packages:

* ``pip install pyphasefield``: Installs pyphasefield, along with required dependencies
* Optional: ``conda install -c conda-forge pycalphad==0.8.4``: Installs pycalphad, along with required dependencies (currently requires an older version, working to update to the latest)

If your computer has a NVIDIA GPU, it is strongly recommended to also install numba (along with cudatoolkit) to be able to run 
phase field simulations on your GPU, as it can lead to speedups of over 100x. To do so, run the following commands:

* Optional: ``conda install numba``: Installs numba, along with required dependencies
* Optional: ``conda install cudatoolkit``: Installs cudatoolkit, which numba needs to make CUDA calls to the GPU

.. warning::
	If you are using an Anaconda environment, Jupyter notebook may not detect packages installed in the environment by default. To ensure 
	this does not lead to "Module not found" errors, run the following commands to install the packages in the current environment into 
	a new kernel for Jupyter notebook (`Source <https://stackoverflow.com/questions/33960051/unable-to-import-a-module-from-python-notebook-in-jupyter>`_):
	
	* ``conda install notebook ipykernel``
	* ``ipython kernel install --user``






.. _PyPI: https://pypi.org/
.. _Anaconda: https://www.anaconda.com/products/individual
.. _pycalphad: https://pycalphad.org/docs/latest/
.. _numba: http://numba.pydata.org/
.. _`Anaconda environment`: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html