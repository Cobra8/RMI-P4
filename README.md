# RMI for P4

This is an adaptation of the original reference recursive model indexes (RMIs) implementation, specifically for generating P4 source code files. A prototype RMI was initially described in [The Case for Learned Index Structures](https://arxiv.org/abs/1712.01208) by Kraska et al. in 2017. The original reference implementation that generates C++ source code files can be found at [learnedsystems/RMI](https://github.com/learnedsystems/RMI).

## Thesis and scope

This work is at best a proof of concept and by no means close to being a viable solution for actual real world switches. A detailed analysis of existing limitations and a more in depth description of my work can be found in my [bachelor thesis](https://github.com/Cobra8/bachelor-thesis).

With that said this implementation focuses on generating P4 source code files that are tested and meant to be used with the [BMv2](https://github.com/p4lang/behavioral-model) reference software switch. In that sense this implementation has a lot more freedom when looking at performance limitations or similar things that real world switches would restrain a lot more.

## Using this implementation

To use this implementation, clone this repository and [install Rust](https://rustup.rs/).

The RMI-P4 implementation is a *compiler* just as the reference implementation is. It takes a dataset as input, and produces P4 and Python source files as outputs. The data input file must be a binary file containing:

1. The number of items, as a 64-bit unsigned integer (little endian)
2. The data items, either 32-bit or 64-bit unsigned integers (little endian)

If the input file contains 32-bit integers, the filename must end with `uint32`. If the input file contains 64-bit integers, the filename must end with `uint64`.

In addition to the input dataset, you must also provide a model structure. For example, to build a 2-layer RMI on the data file `books_200M_uint32` (available from [the Harvard Dataverse](https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/JGVF9A/MZZUP2&version=4.0)) with a branching factor of 100, one could run:

```
cargo run --release -- books_200M_uint32 p4-rmi linear,linear 100 --data-path path/where/to/store/model/parameters
```

Logging useful diagnostic information can be enabled by setting the `RUST_LOG` environmental variable to `trace`: `export RUST_LOG=trace`.

## Using the generated code
The RMI generator produces a P4 source code file and a Python source code file in the current directory. The P4 source code file is meant to be compiled with a P4 compiler such as [p4c](https://github.com/p4lang/p4c). Currently the compiled P4 file is then meant to be run on the [BMv2](https://github.com/p4lang/behavioral-model) switch. Simply copy the generated P4 file in a corresponding Mininet folder and type `make`. As a final step the generated Python source file has to be executed to send the necessary model parameters to the switch using [P4Runtime](https://github.com/p4lang/p4runtime).

To install [BMv2](https://github.com/p4lang/behavioral-model) and the [p4c](https://github.com/p4lang/p4c) compiler, follow the instructions given in the official [P4 Tutorial](https://github.com/p4lang/tutorials) in the section *Obtaining required software*.

## RMI Layers and Tuning

Currently, the following types of RMI layers are supported for P4:

* `linear`, simple linear regression
* `cubic`, connected cubic spline segments
* `radix`, eliminates common prefixes and returns a fixed number of significant bits based on the branching factor

The following remaining types of RMI layers are *not (yet) supported* for P4:

* `normal`, normal CDF with tuned mean, variance, and scale.
* `loglinear`, simple linear regression with a log transform
* `lognormal`, normal CDF with log transform
* `bradix`, same as radix, but attempts to choose the number of bits based on balancing the dataset
* `histogram`, partitions the data into several even-sized blocks (based on the branching factor)
