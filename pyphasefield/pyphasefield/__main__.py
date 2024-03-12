import os
import importlib

success = '\033[92m'
warning = '\033[93m'
fail = '\033[91m'
end_color = '\033[0m'
LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

required_packages = ["numpy", "scipy", "sympy", "matplotlib", "tinydb", "meshio", "symengine"]
pycalphad_prerequisites = ["cython", "numexpr", "bottleneck", "setuptools_scm", "importlib-metadata", "importlib-resources", "pytest", "pytest_cov", "xarray"]

def install_package(name):
    #dont check if package is already installed, it could get confused and not install when it needs to
    print("Attempting to install "+name+" using pip", end='\r')
    out = os.system("python -m pip install "+name+" >> pyphasefield_installation.log 2>&1")
    if(out != 0):
        print(warning+"Failed to install "+name+" using pip, attempting to install with conda"+end_color, end='\r')
        out = os.system("conda install -c conda-forge -y "+name+" >> pyphasefield_installation.log 2>&1")
        if(out != 0):
            print(LINE_CLEAR+fail+"Failed to install "+name+" using conda, unable to proceed!"+end_color)
        else:
            print(LINE_CLEAR+success+"Successfully installed "+name+" using conda!"+end_color)
    else:
        print(success+"Successfully installed "+name+" using pip!"+end_color)
            
coral = input(warning+"Are you running pyphasefield on a CORAL-architecture-supercomputer (POWER9/Tesla GPU) or equivalent? (y/[n]) "+end_color)

if(coral.lower() == "y" or coral.lower() == "yes"):
    coral = True
    jupyter = "y"
    mpi = "y"
    hdf5 = "y"
    cuda = "y"
    ver = "11.4.2"
    calphad = "y"
else:
    coral = False
    cuda = input(warning+"Would you like to install numba and cudatoolkit for GPU simulations? (Requires installation through conda!) (y/[n]) "+end_color)
    if(cuda.lower() == "y" or cuda.lower() == "yes"):
        ver = input(warning+"    Specify version of cudatoolkit here (e.g. 11.4.2), in case the latest is not supported. Leave blank for latest. "+end_color)
    jupyter = input(warning+"Would you like to set up a jupyter notebook kernel? (y/[n]) "+end_color)
    mpi = input(warning+"Would you like to install mpi4py for parallel simulations? (requires MPI to be pre-installed on the supercomputer!) (y/[n]) "+end_color)
    hdf5 = input(warning+"Would you like to build h5py against a parallel installation for running parallel sims? (requires hdf5-parallel to already be installed on the supercomputer!) (y/[n]) "+end_color)
    calphad = input(warning+"Would you like to install pycalphad for calphad thermodynamics integration? (y/[n]) "+end_color)
    
if(jupyter.lower() == "y" or jupyter.lower() == "yes"):
    jupyter = True
else:
    jupyter = False
if(mpi.lower() == "y" or mpi.lower() == "yes"):
    mpi = True
else:
    mpi = False
if(hdf5.lower() == "y" or hdf5.lower() == "yes"):
    hdf5 = True
else:
    hdf5 = False
if(cuda.lower() == "y" or cuda.lower() == "yes"):
    cuda = True
else:
    cuda = False
if(calphad.lower() == "y" or calphad.lower() == "yes"):
    calphad = True
else:
    calphad = False
    
if(cuda):
    print("Attempting to install numba and cudatoolkit, "+warning+"will fail if not run in a conda environment!"+end_color)
    out = os.system("conda install -c conda-forge -y numba >> pyphasefield_installation.log 2>&1")
    if(out != 0):
        print(fail+"Failed to install numba automatically, install this package manually if you would like GPU simulations!"+end_color)
    else:
        print(success+"Successfully installed numba!"+end_color, end='\r')
    if(len(ver) == 0):
        out = os.system("conda install -y -c conda-forge cudatoolkit >> pyphasefield_installation.log 2>&1")
    else:
        out = os.system("conda install -y -c conda-forge cudatoolkit=="+ver+" >> pyphasefield_installation.log 2>&1")
    if(out != 0):
        print(fail+"\nFailed to install cudatoolkit automatically, install this package manually if you would like GPU simulations!"+end_color)
    else:
        print(LINE_UP+LINE_CLEAR+success+"Successfully installed numba!"+end_color)
        print(success+"Successfully installed cudatoolkit!"+end_color)
else:
    print("Skipping installing numba and cudatoolkit")


for name in required_packages:
    install_package(name)
    

    


if(jupyter):
    install_package("ipykernel")
    out = os.system("python -m ipykernel install --prefix=$HOME/.local/ --name 'pyphasefield-jupyter-kernel' --display-name 'Pyphasefield Jupyter Kernel' >> pyphasefield_installation.log 2>&1")
    if (out != 0):
        print(fail+"Failed to install jupyter notebook kernel"+end_color)
    else:
        print(success+"Successfully installed jupyter notebook kernel!"+end_color)
else:
    print("Skipping installing jupyter notebook kernel")

#add LD_PRELOAD environment variable to jupyter kernel if running on a supercomputer, to avoid issues with older libstdc++
if(coral):
    try:
        import json
        original_dir = os.getcwd()
        os.chdir(os.environ["HOME"]+"/.local/share/jupyter/kernels/pyphasefield-jupyter-kernel")
        
        filename = 'kernel.json'
        with open(filename, 'r') as f:
            data = json.load(f)
            data['env'] = {}
            data['env']["LD_PRELOAD"] = os.environ["CONDA_PREFIX"]+"/lib/libstdc++.so"

        os.remove(filename)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(success+"Successfully edited "+os.getcwd()+"/kernel.json to load the correct libstdc++.so library!"+end_color)
        os.chdir(original_dir)
    except:
        print(fail+"Could not edit $HOME/.local/share/jupyter/kernels/pyphasefield-jupyter-kernel/kernel.json, jupyter notebook may not run correctly!"+end_color)
        
    

if(mpi):
    print("Attempting to install mpi4py", end='\r')
    out = os.system('module load gcc >> pyphasefield_installation.log 2>&1; MPICC="mpicc -shared" python -m pip install --no-cache-dir --no-binary=mpi4py mpi4py >> pyphasefield_installation.log 2>&1')
    if(out != 0):
        print(fail+"Failed to install mpi4py automatically, install this package manually if you would like parallel simulations!"+end_color)
    else:
        print(success+"Successfully installed mpi4py!"+end_color)
else:
    print("Skipping installing mpi4py")

if(calphad):
    print("Attempting to install pycalphad using pip", end='\r')
    out = os.system("python -m pip install pycalphad >> pyphasefield_installation.log 2>&1")
    if(out != 0):
        print(warning+"Failed to install pycalphad from prebuilt wheels, attempting to build from source"+end_color)
        print("Installing pre-requisite packages for building from source")
        for name in pycalphad_prerequisites:
            install_package(name)
        print("Attempting to build pycalphad from source", end='\r')
        out = os.system("python -m pip install --no-dependencies --no-build-isolation git+http://github.com/pycalphad/pycalphad.git >> pyphasefield_installation.log 2>&1")
        if(out != 0):
            print(fail+"Failed to install pycalphad automatically, install this package manually if you would like calphad-based simulations!"+end_color)
        else:
            print(success+"Successfully installed pycalphad from source!"+end_color)
    else:
        print(success+"Successfully installed pycalphad using pip!"+end_color)
else:
    print("Skipping installing pycalphad")
        

if(hdf5):
    print("Attempting to install h5py against a pre-built parallel installation", end='\r')
    out = os.system("""module load gcc/8 >> pyphasefield_installation.log 2>&1; 
                       module load hdf5-parallel >> pyphasefield_installation.log 2>&1;
                       export CC=mpicc >> pyphasefield_installation.log 2>&1;
                       export CXX=mpicxx >> pyphasefield_installation.log 2>&1;
                       conda install -c conda-forge -y cython >> pyphasefield_installation.log 2>&1;
                       conda install -c conda-forge -y pkgconfig >> pyphasefield_installation.log 2>&1;
                       export HDF5_DIR=$HDF5 >> pyphasefield_installation.log 2>&1;
                       pip install --force --no-dependencies --no-binary=h5py h5py >> pyphasefield_installation.log 2>&1;
                       """)
    if(out != 0):
        print(fail+"Failed to install h5py automatically, install this package manually if you would like parallel IO!"+end_color)
    else:
        print(success+"Successfully installed h5py!"+end_color)
else:
    print("Attempting to install h5py against a serial installation, parallel IO will not work!", end='\r')
    out = os.system("""module load gcc/8 >> pyphasefield_installation.log 2>&1; 
                       pip install h5py;
                       """)
    if(out != 0):
        print(LINE_CLEAR+fail+"Failed to install h5py automatically, install this package manually in order to save data!"+end_color)
    else:
        print(LINE_CLEAR+success+"Successfully installed h5py!"+end_color)
    
    
