# ITensorContractionOrders

This package ports a state of the art contraction order finding package `OMEinsumContractionOrders` to `ITensors` to enable large scale tensor network contraction.

[![CI](https://github.com/GiggleLiu/ITensorContractionOrders.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/GiggleLiu/ITensorContractionOrders.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/GiggleLiu/ITensorContractionOrders.jl/branch/main/graph/badge.svg?token=ZkoPZJeW2v)](https://codecov.io/gh/GiggleLiu/ITensorContractionOrders.jl)

## Installation
<p>
ITensorContractionOrders is a &nbsp;
    <a href="https://julialang.org">
        <img src="https://raw.githubusercontent.com/JuliaLang/julia-logo-graphics/master/images/julia.ico" width="16em">
        Julia Language
    </a>
    &nbsp; package. To install ITensorContractionOrders,
    please <a href="https://docs.julialang.org/en/v1/manual/getting-started/">open
    Julia's interactive session (known as REPL)</a> and press <kbd>]</kbd> key in the REPL to use the package mode, then type
</p>

```julia
pkg> add ITensorContractionOrders
```

To update, just type `up` in the package mode.

## Examples

```julia
julia> using ITensors, ITensorContractionOrders

julia> i, j, k, l = Index(4), Index(5), Index(6), Index(7)
((dim=4|id=297), (dim=5|id=593), (dim=6|id=803), (dim=7|id=187))

julia> x, y, z = randomITensor(i, j), randomITensor(j, k), randomITensor(k, l);

julia> (tc, sc, rw) = contraction_complexity(net)
Time complexity (number of element-wise multiplications) = 2^8.169925001442312
Space complexity (number of elements in the largest intermediate tensor) = 2^4.807354922057604
Read-write complexity (number of element-wise read and write) = 2^7.39231742277876
```

## Supporting and Citing

Much of the software in this ecosystem was developed as part of academic research.
If you would like to help support it, please star the repository as such metrics may help us secure funding in the future.
If you use our software as part of your research, teaching, or other activities, we would be grateful if you could cite our work.
The
[CITATION.bib](CITATION.bib) file in the root of this repository lists the relevant papers.

## Acknowledgement
* Helpful discussion with @mtfishman:
https://github.com/ITensor/ITensors.jl/pull/954
