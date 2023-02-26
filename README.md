# HotSpot-GPU

This is a GPU accelerated version of the HotSpot 5.02 architecture level thermal 
modeling infrastructure originally developed at the University of Virginia.

Most of the GPU code was written for [an old project in
2014](https://bitbucket.org/huanzhang/hotspot-gpu/), but surprisingly
everything still works with minor fixes in 2023. The GPU code was written in
OpenCL and was originally tuned for AMD GCN architecture (achieved 90X
speedup), but it also shows 70 - 100X speedup on NVIDIA Tesla V100 GPUs in my
latest benchmark in 2023.

## Build Instructions:

First you will need to install OpenCL driver from GPU vendor. For NVIDIA cards
on Ubuntu you can run:

```
sudo apt install nvidia-driver-525
```

For AMD cards, the AMDGPU-Pro or ROCm driver should work.  Besides the OpenCL
driver provided by GPU vendor, these OpenCL packages are needed for providing
OpenCL headers and runtime libraries:

```
sudo apt install opencl-headers ocl-icd-opencl-dev ocl-icd-libopencl1
```

To compile with OpenCL support, please run:

```
make GPGPU=1
```

Double precision kernel will be used by default. For OpenCL platforms that do
not support double precision or have poor double precision performance (such as
most NVIDIA GTX/RTX series cards), you can compile the single precision kernel:

```
make GPGPU=1 GPGPU_SINGLE=1
```

Usually the difference between single and double precision results will be
below 0.01 degree, but please expect larger errors if you have a large
number of simulation time steps.

## How to Use: ##

The following new command line switches have been added:

* -gpu_enable (Default: 0, disabled)
* -gpu_device (Default: 0)
* -gpu_platform (Default: 0)

To enable GPU acceleration on OpenCL platform 0 device 0, add the following 
three new command line switches to HotSpot:

```
-gpu_enable 1 -gpu_platform 0 -gpu_device 0
```

## Tested Platforms: ##

* NVIDIA Tesla V100 with 525.78.01 driver (tested 2023)
* Radeon HD 7970 GHz Edition and R9 280X with fglrx-14.301 driver (tested 2014)
* Intel(R) HD Graphics IvyBridge M GT2 with beignet 1.0.0 (tested 2014)
* Intel(R) HD Graphics Haswell GT2 Desktop with beignet 1.0.0 (tested 2014)

2014 benchmark results: Comparing to a 2.6 GHz IvyBrige CPU, HD 7970 GHz
Edition provides about 90X speedup when the grid dimension is 1024X1024 double
precision. Using single precision may give you almost doubled performance.  The
two Intel integrated graphic cards listed above provide over 20X speedup at
1024X1024 single precision.

Command used for benchmarking:

```
./hotspot -c hotspot.config -init_file gcc.init -f ev6.flp -p gcc.ptrace -o gcc.ttrace -model_type grid -grid_rows 512 -grid_cols 512 -gpu_enable 1 -gpu_platform 0 -gpu_device 0
```

2023 benchmark results: The command took **1.25s** on Tesla V100 (double
precision), and **90.3s** on Xeon Gold 6138 (about **72X speedup**). Generated
gcc.ttrace files are exactly the same.  With single precision, the GPU version
took **0.85s** (over **100X speedup**) and the simulation difference is up to
0.01 degree.

Note that most consumer grade NVIDIA GPUs (RTX/GTX) have very poor
double-precision performance due to market segmentation reasons. So using
single precision might be a better option if you don't have access to
data-center GPUs (such as V100, A100). AMD's consumer grade GPUs work much
better for double precision computation compared to NVIDIA's.
