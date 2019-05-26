# HYPHA (A Hybrid Persistent Homology matrix reduction Accelerator)
Copyright 2018, 2019 The Ohio State University 

Simon Zhang, Mengbai Xiao, Chengxin Guo, Liang Geng, Hao Wang, Xiaodong Zhang

HYPHA makes calls to the PHAT library `[3]`, developed by:
Ulrich Bauer, Michael Kerber, Jan Reininghaus, Hubert Wagner

## Description

HYPHA is a hybrid computing engine executed by both GPU and multicore to achieve high performance for computing persistent homology matrix reduction `[6]`.

The essential foundation of our algorithm design and implementation is the separation of SIMT and MIMD parallelisms in Persistent Homology (PH) matrix reduction computation. With such a separation, we are able to perform massive parallel scanning operations on GPU in a super-fast manner, which also collects rich information from an input boundary matrix for further parallel reduction operations on multicore with high efficiency.

For detailed information about the framework and algorithms design of HYPHA, please refer to the paper in ACM ICS 2019 `[8]`.

## Installation

Dependencies: 

1. Make sure cmake is installed at the correct version for the cmake file (e.g. cmake 2.8.12.2)

2. Make sure CUDA is installed at the correct version for the cmake file (e.g. CUDA 10.0.130)

3. Make sure GCC is installed at the correct version for the cmake file (e.g. GCC 7.3.0)

HYPHA is intended to run on high performance computing systems

Thus, a GPU with enough device memory is needed to run large datasets. (e.g. Tesla V100 GPU with 16GB device DRAM). If the system's GPU is not compatible, or the system does not have a GPU, error messages will appear.

Furthermore, it is also preferable to have a multicore processor (e.g. >= 28 cores) for effective computation, and a large amount of DRAM is required for large datasets. We have tested on a 192 GB DRAM single computing node with 40 cores.

If Unix-based operating system is used, the following actions are needed for installation:

type the following sequence of commands:

```
git clone https://github.com/MandMs/hypha.git
cd hypha
mkdir build
cd build && cmake .. && make
```
## To Run the Given Example

After running make from the build folder, first untar the examples folder

```
cd ..
tar -xvf examples.tar.gz
cd build
```

then run the command:

```
./hypha --spectral_sequence ../examples/high_genus_extended.bin high_genus_extended-pairs.res
```

In general, to run in the build folder, type:

```
./hypha [options] inputfile outputfile
```

where inputfile and outputfile have paths relative to the build folder

options:

```
Usage: hypha [options] input_filename output_filename

Options:

--ascii         --  use ascii file format
--binary        --  use binary file format (default)
--help          --  prints this screen
--verbose       --  verbose is always on

--dual_input    --  convert pairs to persistent homology pairs from reduction of a dualized input boundary matrix
--nocompression -- removes compression (may be beneficial with dualized input)

--vector_vector, --full_pivot_column, --sparse_pivot_column, --heap_pivot_column, --bit_tree_pivot_column
--  selects a representation data structure for boundary matrices (default is '--bit_tree_pivot_column')

--twist, --spectral_sequence
--  selects a final stage matrix reduction algorithm (default is '--twist')

```

CAUTION: if the wrong combination of options are typed, HYPHA's behavior will be undefined or it may recognize the right most option or will print the above help screen.

## For Datasets

HYPHA is fully compatible with PHAT's input boundary matrix binary and ascii inputs as well as the binary and ascii persistence pairs output files. See `[3]` https://github.com/blazs/phat for a description of the binary and ascii inputs.

See the PHAT project's benchmark datasets here:

https://drive.google.com/uc?id=0B7Yz6TPEpiGERGZFbjlXaUt1ZWM&export=download

To generate datasets see the project by Bauer, Kerber, Reininghaus:

`[2]` https://github.com/DIPHA/dipha

Notice that DIPHA may require a lower version of openmpi than currently installed on the cluster. e.g. openmpi version 3.0.1

In the dipha/matlab folder, there are matlab scripts to create a .complex file

e.g. for a large dataset: ```create_smooth_image_data(100)``` will generate smooth_100.complex

Then in the build folder, run ```./create_phat_filtration ../matlab/inputfile.complex outputfile.bin```

e.g. run ```./create_phat_filtration ../matlab/smooth_100.complex smooth_100.bin```

then move the .bin file to your examples folder under HYPHA

to generate anti-transposed (dualized) matrices `[7]`, turn on the --dual option when running ./create_phat_filtration

e.g. run ```./create_phat_filtration --dual ../matlab/smooth_100.complex smooth_100-DUAL.bin```

## Using the `convert` Executable

To convert datasets from/to binary to/from ascii and with/without dualizing, use the convert executable available in the build folder upon successfully building.

e.g. run ```./convert --binary --save-binary --dualize inputfile.bin inputfileDUAL.bin```

Note: In special cases `[4]`, reducing a dualized matrix can significantly reduce the amount of computation needed and thus may give better performance with the --nocompression option on. HYPHA performs compression `[1]` by default. 

## Other PH Softwares of Interest

Here is a brief list of related PH computation software packages: (ordered alphabetically)

[Dionysus](https://github.com/mrzv/dionysus)

[DIPHA](https://github.com/DIPHA/dipha)

[Eirene](https://github.com/Eetion/Eirene.jl)

[GUDHI](http://gudhi.gforge.inria.fr/)

[JavaPlex](https://github.com/appliedtopology/javaplex)

[Perseus](http://people.maths.ox.ac.uk/nanda/perseus/index.html)

[PHAT](https://github.com/blazs/phat)

[Ripser](https://github.com/Ripser/ripser)

[TDA](https://cran.r-project.org/web/packages/TDA/index.html)

## References

  1. U. Bauer, M. Kerber, J. Reininghaus: _Clear and Compress: Computing Persistent Homology in Chunks_. [http://arxiv.org/pdf/1303.0477.pdf arXiv:1303.0477]
  2. U. Bauer, M. Kerber, J. Reininghaus: [Distributed computation of persistent homology](http://dx.doi.org/10.1137/1.9781611973198.4). Proceedings of the Sixteenth Workshop on Algorithm Engineering and Experiments (ALENEX), 2014.
  3. U. Bauer, M. Kerber, J. Reininghaus, H. Wagner: [PHAT – Persistent Homology Algorithms Toolbox](https://people.mpi-inf.mpg.de/~mkerber/bkrw-pphat.pdf). Mathematical Software – ICMS 2014, Lecture Notes in Computer Science Volume 8592, 2014, pp 137-143
  4. U. Bauer, “Ripser/Ripser.” GitHub, 6 Oct. 2018, github.com/Ripser/ripser.
  5. C. Chen, M. Kerber: _Persistent Homology Computation With a Twist_. 27th European Workshop on Computational Geometry, 2011.
  6. H. Edelsbrunner, J. Harer: _Computational Topology, An Introduction_. American Mathematical Society, 2010, ISBN 0-8218-4925-5
  7. V. de Silva, D. Morozov, M. Vejdemo-Johansson: _Dualities in persistent (co)homology_. Inverse Problems 27, 2011
  8. S. Zhang, M. Xiao, C. Guo, L. Geng, H. Wang, X. Zhang: HYPHA: a Framework based on Separation of Parallelisms to Accelerate Persistent Homology Matrix Reduction. Proceedings of the 2019 International Conference on Supercomputing. ACM, 2019.

