:orphan:

###############
Getting started
###############

Installation
============

Qiskit Nature PySCF depends on the main Qiskit package which has its own
`installation instructions <https://quantum.cloud.ibm.com/docs/guides/install-qiskit>`__ detailing the
installation options for Qiskit and its supported environments/platforms. You should refer to
that first. Then the information here can be followed which focuses on the additional installation
specific to Qiskit Nature PySCF.

.. tab-set::

    .. tab-item:: Start locally

        .. code:: sh

            pip install qiskit-nature-pyscf


    .. tab-item:: Install from source

       Installing Qiskit Nature PySCF from source allows you to access the most recently
       updated version under development instead of using the version in the Python Package
       Index (PyPI) repository. This will give you the ability to inspect and extend
       the latest version of the Qiskit Nature PySCF code more efficiently.

       Since Qiskit Nature PySCF depends on Qiskit, and its latest changes may require new or changed
       features of Qiskit, you should first follow Qiskit's `"Install from source"` instructions
       `here <https://quantum.cloud.ibm.com/docs/guides/install-qiskit-source>`__

       .. raw:: html

          <h2>Installing Qiskit Nature PySCF from Source</h2>

       Using the same development environment that you installed Qiskit in you are ready to install
       Qiskit Nature PySCF.

       1. Clone the Qiskit Nature PySCF repository.

          .. code:: sh

             git clone https://github.com/qiskit-community/qiskit-nature-pyscf.git

       2. Cloning the repository creates a local folder called ``qiskit-nature-pyscf``.

          .. code:: sh

             cd qiskit-nature-pyscf

       3. If you want to run tests or linting checks, install the developer requirements.

          .. code:: sh

             pip install -r requirements-dev.txt

       4. Install ``qiskit-nature-pyscf``.

          .. code:: sh

             pip install .

       If you want to install it in editable mode, meaning that code changes to the
       project don't require a reinstall to be applied, you can do this with:

       .. code:: sh

          pip install -e .

----


.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`
