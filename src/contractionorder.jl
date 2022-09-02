# Port OMEinsumContractionOrders to ITensors
# Slicing is not supported, because it might require extra work to slice an `ITensor` correctly.

const ITensorList = Union{Vector{<:ITensor},Tuple{Vararg{<:ITensor}}}

"""
$(TYPEDEF)
    ITensorNetwork(args) -> ITensorNetwork

Define a tensor network with its contraction order specified by a tree structure.
In this network, each index in this tensor network must appear either twice or once.
The input `args` is a Vector of [`ITensor`](@ref) or another layer of Vector.
This data type can be automatically generated from [`optimize_contraction`](@ref) function.

### Fields
$(TYPEDFIELDS)

### Examples

The following code creates a tensor network and evaluates it in a sequencial order.

```jldoctest
julia> using ITensors, ITensorContractionSequences

julia> i, j, k, l = Index(4), Index(5), Index(6), Index(7);

julia> x, y, z = randomITensor(i, j), randomITensor(j, k), randomITensor(k, l);

julia> it = ITensorNetwork([[x, y] ,z]);

julia> itensor_list = ITensorContractionSequences.flatten(it);  # convert this tensor network to a Vector of ITensors

julia> evaluate(it) ≈ foldl(*, itensor_list)
true
```
"""
struct ITensorNetwork{IT}
    args::Vector{Union{ITensorNetwork, ITensor}}
    iy::Vector{Index{IT}}   # the output labels, note: this is type unstable
end
ITensors.inds(it::ITensorNetwork) = (it.iy...,)

function ITensorNetwork(args)::ITensorNetwork
    args = Union{ITensorNetwork, ITensor}[arg isa Union{AbstractVector, Tuple} ? ITensorNetwork(arg) : arg for arg in args]
    # get output labels
    # NOTE: here we assume the output index id has `Int` type
    labels = collect.(Index{Int}, ITensors.inds.(args))
    return ITensorNetwork(args, infer_output(labels))
end

"""
    flatten(it::ITensorNetwork) -> Vector

Convert an [`ITensorNetwork`](@ref) to a Vector of [`ITensor`](@ref).
"""
flatten(it::ITensorNetwork) = flatten!(it, ITensor[])
function flatten!(it::ITensorNetwork, lst)
    for arg in it.args
        if arg isa ITensor
            push!(lst, arg)
        else
            flatten!(arg, lst)
        end
    end
    return lst
end

# Contract and evaluate an itensor network.
"""
$(TYPEDSIGNATURES)
"""
evaluate(it::ITensorNetwork)::ITensor = foldl(*, evaluate.(it.args))
evaluate(it::ITensor) = it

############################ Port to OMEinsumContractionOrders #######################
getid(index::Index) = index
getids(A::ITensor) = Index{Int}[getid(x) for x in ITensors.inds(A)]
getids(A::ITensorNetwork) = collect(Index{Int}, getid.(ITensors.inds(A)))
function rootcode(it::ITensorNetwork)
    ixs = [getids(A) for A in it.args]
    return OMEinsumContractionOrders.EinCode(ixs, it.iy)
end

function update_size_index_dict!(size_dict::Dict{Index{IT}}, index_dict::Dict{Index{IT}}, tensor::ITensor) where IT
    for ind in ITensors.inds(tensor)
        size_dict[getid(ind)] = ind.space
        index_dict[getid(ind)] = ind
    end
    return size_dict
end

# decorate means converting the raw contraction pattern to ITensorNetwork.
# `tensors` is the original input tensor list.
function decorate(net::OMEinsumContractionOrders.NestedEinsum, tensors::ITensorList)
    if OMEinsumContractionOrders.isleaf(net)
        return tensors[net.tensorindex]
    else
        return ITensorNetwork(decorate.(net.args, Ref(tensors)))
    end
end

# get a (labels, size_dict) representation of a ITensorNetwork
function rawcode(tensors::ITensorList)
    # we use id as the label
    indsAs = [collect(Index{Int}, ITensors.inds(A)) for A in tensors]
    ixs = [getids(x) for x in tensors]
    unique_labels = unique(vcat(indsAs...))
    size_dict = Dict([getid(x)=>x.space for x in unique_labels])
    index_dict = Dict([getid(x)=>x for x in unique_labels])
    return OMEinsumContractionOrders.EinCode(ixs, getid.(infer_output(indsAs))), size_dict, index_dict
end

# infer the output tensor labels
function infer_output(inputs::AbstractVector{<:AbstractVector{Index{IT}}}) where IT
    indslist = vcat(inputs...)
    # get output indices
    iy = Index{IT}[]
    for l in indslist
        c = count(==(l), indslist)
        if c == 1
            push!(iy, l)
        elseif c !== 2
            error("Each index in a tensor network must appear at most twice!")
        end
    end
    return iy
end

function rawcode(net::ITensorNetwork{IT}) where IT
    size_dict = Dict{Index{IT},Int}()
    index_dict = Dict{Index{IT},Index{Int}}()
    r = rawcode!(net, size_dict, index_dict)
    return r, size_dict, index_dict
end
function rawcode!(net::ITensorNetwork{IT}, size_dict::Dict{Index{IT}}, index_dict::Dict{Index{IT}}, index_counter=Base.RefValue(0)) where IT
    args = map(net.args) do s
        if s isa ITensor
            update_size_index_dict!(size_dict, index_dict, s)
            OMEinsumContractionOrders.NestedEinsum{Index{IT}}(index_counter[] += 1)
        else  # ITensorNetwork
            scode = rawcode!(s, size_dict, index_dict, index_counter)
            # no need to update size, size is only updated on the leaves.
            scode
        end
    end
    return OMEinsumContractionOrders.NestedEinsum(args, rootcode(net))
end

"""
$(TYPEDSIGNATURES)

Optimize the contraction order of a tensor network specified as a vector tensors.
Returns a [`ITensorNetwork`](@ref) instance, which can be evaluated with the [`evaluate`](@ref) function.

### Examples
```jldoctest
julia> using ITensors, ITensorContractionSequences

julia> i, j, k, l = Index(4), Index(5), Index(6), Index(7);

julia> x, y, z = randomITensor(i, j), randomITensor(j, k), randomITensor(k, l);

julia> net = optimize_contraction([x, y, z]; optimizer=TreeSA());
```
"""
function optimize_contraction(tensors::ITensorList; optimizer::CodeOptimizer=TreeSA())::ITensorNetwork
    r, size_dict, index_dict = rawcode(tensors)
    # merge vectors can speed up contraction order finding
    # optimize the permutation of tensors is set to true
    res = optimize_code(r, size_dict, optimizer, MergeVectors(), true)
    if res isa OMEinsumContractionOrders.SlicedEinsum   # slicing is not supported!
        if length(res.slicing) != 0
            @warn "Slicing is not yet supported by `ITensors`, removing slices..."
        end
        res = res.eins
    end
    return decorate(res, tensors)
end

"""
    peak_memory(net::ITensorNetwork) -> Int
"""
OMEinsumContractionOrders.peak_memory(net::ITensorNetwork) = peak_memory(rawcode(net)[1:2]...)

"""
    flop(net::ITensorNetwork) -> Int
"""
OMEinsumContractionOrders.flop(net::ITensorNetwork) = flop(rawcode(net)[1:2]...)

"""
    timespacereadwrite_complexity(net::ITensorNetwork) -> (tc, sc, rwc)
"""
OMEinsumContractionOrders.timespacereadwrite_complexity(net::ITensorNetwork) = timespacereadwrite_complexity(rawcode(net)[1:2]...)

"""
    timespace_complexity(net::ITensorNetwork) -> (tc, sc)
"""
OMEinsumContractionOrders.timespace_complexity(net::ITensorNetwork) = timespacereadwrite_complexity(rawcode(net)[1:2]...)[1:2]

"""
    label_elimination_order(net::ITensorNetwork) -> Vector
"""
function OMEinsumContractionOrders.label_elimination_order(net::ITensorNetwork)
    r, size_dict, index_dict = rawcode(net)
    return getindex.(Ref(index_dict), label_elimination_order(r))
end