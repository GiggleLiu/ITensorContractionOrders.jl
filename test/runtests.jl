using ITensors
using ITensorContractionOrders
using Test, Documenter

@testset "contractionorder.jl" begin
  include("contractionorder.jl")
end

@testset "doctest" begin
  Documenter.doctest(ITensorContractionOrders; manual=false)
end