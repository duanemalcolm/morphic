************
Installation
************

.. toctree::
   :maxdepth: 2


The only way to install Morphic at the moment is to download the source
from Git Hub. In the future, we plan to create an easy_install Python
package.

============
Requirements
============
    - numpy
    - scipy
    
--------
Optional
--------
    - sparsesvd for fitting
    - matplotlib 2D plotting
    - mayavi2 for 2D and 3D plotting

============
From Git Hub
============
Clone the Morphic code::

    git clone http://github.com/duanemalcolm/morphic

Add the morphic path to the Python path by adding it to your shell script::

    nano ~/.bashrc
    export PYTHONPATH=$PYTHONPATH:~/path/to/morphic

Run an example::

    cd ~/path/to/morphic/examples
    python example_1d_linear.py plot
    
==============
From a Package
==============

.. note::
    
    We plan to create easy install packages in the future
