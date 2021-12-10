# RMI for P4

This is an implementation of the recursive model indexes (RMIs) specifically for generating P4 code. A prototype RMI was initially described in [The Case for Learned Index Structures](https://arxiv.org/abs/1712.01208) by Kraska et al. in 2017.

![Fig 1 from the Case for Learned Index Structures](http://people.csail.mit.edu/ryanmarcus/rmi.png)

## The Original

The original reference implementation that generates CPP code can be found at [learnedsystems/RMI](https://github.com/learnedsystems/RMI).

## Using this implementation

To use this implementation, clone this repository and [install Rust](https://rustup.rs/).

The reference RMI implementation is a *compiler.* It takes a dataset as input, and produces P4 source files as outputs. The data input file must be a binary file containing:

1. The number of items, as a 64-bit unsigned integer (little endian)
2. The data items, either 32-bit or 64-bit unsigned integers (little endian)

If the input file contains 32-bit integers, the filename must end with `uint32`. If the input file contains 64-bit integers, the filename must end with `uint64`. If the input file contains 64-bit floats, the filename must end with `f64`.

In addition to the input dataset, you must also provide a model structure. For example, to build a 2-layer RMI on the data file `books_200M_uint32` (available from [the Harvard Dataverse](https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/JGVF9A/MZZUP2&version=4.0)) with a branching factor of 100, we could run:

```
cargo run --release -- books_200M_uint32 my_first_rmi linear,linear 100
```

Logging useful diagnostic information can be enabled by setting the `RUST_LOG` environmental variable to `trace`: `export RUST_LOG=trace`.


## Generated code
The RMI generator produces a P4 source file in the current directory.
TODO - describe how to use P4 source file briefly here?

## RMI Layers and Tuning

Currently, the following types of RMI layers are supported:

* `linear`, simple linear regression
* `linear_spline`, connected linear spline segments
* `cubic`, connected cubic spline segments
* `loglinear`, simple linear regression with a log transform
* `normal`, normal CDF with tuned mean, variance, and scale.
* `lognormal`, normal CDF with log transform
* `radix`, eliminates common prefixes and returns a fixed number of significant bits based on the branching factor
* `bradix`, same as radix, but attempts to choose the number of bits based on balancing the dataset
* `histogram`, partitions the data into several even-sized blocks (based on the branching factor)

Tuning an RMI is critical to getting good performance. A good place to start is a `cubic` layer followed by a large linear layer, for example: `cubic,linear 262144`. For automatic tuning, try the RMI optimizer using the `--optimize` flag:

```
cargo run --release -- --optimize optimizer_out.json books_200M_uint64
```

By default, the optimizer will use 4 threads. If you have a big machine, consider increasing this with the `--threads` option.

The optimizer will output a table, with each row representing an RMI configuration. By default, the optimizer selects a small set of possible configurations that are heuristically selected to cover the Pareto front. Each column contains:

* `Models`: the model types used at each level of the RMI
* `Branch`: the branching factor of the RMI (number of leaf models)
* `AvgLg2`: the average log2 error of the model (which approximates the number of binary search steps required to find a particular key within a range predicted by the RMI)
* `MaxLg2`: the maximum log2 error of the model (the maximum number of binary search steps required to find any key within the range predicted by the RMI)
* `Size (b)`: the in-memory size of the RMI, in bytes.
