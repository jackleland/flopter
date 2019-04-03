Flopter
=======

Flopter is a python package for the preparation and fitting/plotting of Langmuir probe 
data from a variety of sources. So far there is functionality covering the particle-in-
cell code SPICE-2 (with plans to expand this to SPICE-3) as well as integration with the 
database used at DIFFER.  

Documentation can be found: at http://spicerack.gitpages.ccfe.ac.uk/flopter

Requirements
------------

Flopter currently runs on python 3.6. Current requirements are:

- numpy - 1.16.1
- scipy - 1.1.0
- matplotlib - 2.2.2
- pandas - 0.23.4
- xarray - 0.11.3

The DIFFER database modules, and by extension anything revolving around use of data taken 
on Magnum-PSI data, requires:

- zeroc-ice - 3.7.1
- codac (not currently available on PyPI, will need to contact DIFFER to install and use)  

And for generating the documentation:

- Sphinx - 2.0.0
- sphinx-rtd-theme - 0.4.3

Examples and quick scripts are written in Jupyter notebooks, and so will require the 
jupyter package (or similar) to run. A full list of requirements is included in pip-
friendly format.


Installation
------------

Flopter is not currently on PyPI, so installation is only available through the cloning 
of this repository. To install flopter with all required packages, :

```bash
git clone git@git.ccfe.ac.uk:SpiceRack/flopter.git
pip install -r flopter/requirements.txt
```



Contribution
------------

This repository is written and maintained by Jack Leland (j.leland@liverpool.ac.uk), if 
you would like to contribute then feel free to contact me, otherwise you are very welcome 
to fork and submit changes as a merge request. 