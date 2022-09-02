module ITensorContractionOrders

using ITensors, OMEinsumContractionOrders, DocStringExtensions

# this changes the default warn order!!!!
set_warn_order!(32)

# tensor netork API
export ITensorNetwork, evaluate

# re-export the functions in OMEinsumContractionOrders
export KaHyParBipartite, GreedyMethod, TreeSA, SABipartite,
    MinSpaceDiff, MinSpaceOut,
    MergeGreedy, MergeVectors,
    optimize_contraction,
    # time space complexity
    peak_memory, timespace_complexity, timespacereadwrite_complexity, flop,
    label_elimination_order

include("contractionorder.jl")

end
