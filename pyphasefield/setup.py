import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
with open("VERSION", "r") as v:
    version_string = v.read()

setuptools.setup(
    name="pyphasefield",
    version=version_string,
    author="Scott Peters",
    author_email="scott@dpeters.net",
    description="Python phase field simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AdditiveModeling/pyphasefield",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
	install_requires=[
        'numpy>=1.13, <2.0',
        'pycalphad>=0.11',
        'sympy>=1.12',
    ],
    python_requires='>=3.10',
)
