# Numerical simulation of Ising model on GPU with CUDA

This project is about using driver CUDA to execute the code faster on the GPU than on the CPU.
I simulated the Ising model (a simple model for ferromagnetism in matter) employing a MonteCarlo Mark Chain (MCMC).
The simulation is performed in C (using CUDA driver) and the analysis results are computed in python.
A time comparison between pure C and Cuda will be reported in the future below.

# Installation
----------------------

* Clone this repo to your computer.
* Install the requirements using 'pip install -r requirements.txt'.
    * Make sure you use Python 3.
    * A virtual environment is recommended.
* Install CUDA following the offical [docs](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

# Usage
-----------------------

* Run C code with the following
    * Modify the relevant parameters in lines 7-13 of *dim2.c*, expecially the path the results will be saved in and the lattice spacing
    * Compile with 'gcc dim2.c -o dim2c -lm'
    * Execute *dim2c* to generate result

* Run C code with the following
    * Modify the relevant parameters in lines 11-19 of *dim2.cu*, expecially the path the results will be saved in and the lattice spacing
    * Compile with 'nvcc dim2.cu -o dim2cu -lcurand'
    * Execute *dim2cu* to generate result

* Error analysis at given lattice spacing and temperature
    * Run 'python single_temperature_err_analysis' to compute the mean quantities (energy, magnetization, suscettibility) and associated errors.
        techniques are used to take into account autocorrelation between the measures.
        The scripts runs on files named *data/lattice\*cu.dat*.
* Run 'python multiple_temperature.py' to a comprehesive analysis of mean quantities around phase transition temperature and dependance from latting spacing.
    The scripts runs on files named *data/lattice\*cu.dat*.

# Time Comparison
----------------------
The number of steps between measures is proportional to *lattice*x*lattice*,
so we expect a quadratic (in 2D) increase in computational time
(x4, doubling the lattice points each time).

    lattice     |   C code runtime (s)  |   CUDA code runtime (s)
----------------|-----------------------|-------------------------
        4       |         0.221         |            5.364
        8       |         0.719         |            5.597
        16      |         2.431         |            5.723
        32      |         9.806         |            7.564
        64      |        38.655         |           14.874
        128     |       148.755         |           47.485
        256     |       630.897         |          169.416

We see that for a relatively large *lattice* (>30 ca), the CUDA program is actually faster.


# Licence
----------------------
This work is licenced under (Creative Commons Attribution-NonCommercial 3.0 Unported License)[https://creativecommons.org/licenses/by-nc/3.0/]
by Federico Visintini.
