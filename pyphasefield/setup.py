import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyphasefield",
    version="0.0.10",
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
        "numpy",
		"matplotlib",
		"meshio",
    ],
    python_requires='>=3.6',
)