#!/usr/bin/env julia

using Test
using JuMP
using Alpine
using Ipopt
using Juniper
using MathOptInterface
import HiGHS

const MOI = MathOptInterface
const _HAS_GUROBI = try
    @eval import Gurobi
    true
catch
    false
end

function json_escape(s::AbstractString)
    escaped = replace(s, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n", "\r" => "\\r", "\t" => "\\t")
    return escaped
end

function write_record(io, record::Dict{String, Any})
    parts = String[]
    for key in sort!(collect(keys(record)))
        value = record[key]
        encoded = if value === nothing
            "null"
        elseif value isa Bool
            value ? "true" : "false"
        elseif value isa Integer || value isa AbstractFloat
            string(value)
        else
            "\"" * json_escape(string(value)) * "\""
        end
        push!(parts, "\"" * key * "\":" * encoded)
    end
    write(io, "{" * join(parts, ",") * "}\n")
end

function build_mip_solver(per_instance_time_limit::Float64, mip_backend::AbstractString)
    backend = lowercase(String(mip_backend))
    if backend == "highs"
        return MOI.OptimizerWithAttributes(
            HiGHS.Optimizer,
            "presolve" => "on",
            "log_to_console" => false,
            "time_limit" => per_instance_time_limit,
        )
    end

    if backend == "gurobi"
        _HAS_GUROBI || error("ALPINE_MIP_SOLVER=gurobi requires Gurobi.jl in the active Julia environment")
        grb_env = Gurobi.Env()
        return MOI.OptimizerWithAttributes(
            () -> Gurobi.Optimizer(grb_env),
            MOI.Silent() => true,
            "Presolve" => 1,
            "TimeLimit" => per_instance_time_limit,
        )
    end

    error("unsupported ALPINE_MIP_SOLVER=$(mip_backend); expected 'highs' or 'gurobi'")
end

function build_optimizer(per_instance_time_limit::Float64, mip_backend::AbstractString)
    ipopt = MOI.OptimizerWithAttributes(
        Ipopt.Optimizer,
        MOI.Silent() => true,
        "sb" => "yes",
        "max_iter" => 9999,
        "max_wall_time" => per_instance_time_limit,
    )
    mip_solver = build_mip_solver(per_instance_time_limit, mip_backend)
    juniper = MOI.OptimizerWithAttributes(
        Juniper.Optimizer,
        MOI.Silent() => true,
        "mip_solver" => mip_solver,
        "nl_solver" => ipopt,
    )
    return JuMP.optimizer_with_attributes(
        Alpine.Optimizer,
        "nlp_solver" => ipopt,
        "mip_solver" => mip_solver,
        "minlp_solver" => juniper,
        "time_limit" => per_instance_time_limit,
    )
end

function resolve_symbol_name(symbol_name::AbstractString, minlptests)
    symbol_name_str = String(symbol_name)
    requested = Symbol(symbol_name_str)
    if isdefined(minlptests, requested)
        return symbol_name_str
    end

    match_501 = match(r"^(nlp_cvx_501_01[01])_[0-9]+d$", symbol_name_str)
    if match_501 !== nothing
        canonical_name = match_501.captures[1]
        if isdefined(minlptests, Symbol(canonical_name))
            return canonical_name
        end
    end

    return nothing
end

function missing_symbol_record(problem_id::AbstractString, symbol_name::AbstractString)
    problem_id_str = String(problem_id)
    symbol_name_str = String(symbol_name)
    return Dict(
        "problem_id" => problem_id_str,
        "symbol" => symbol_name_str,
        "outcome" => "error",
        "passes" => 0,
        "fails" => 0,
        "errors" => 1,
        "broken" => 0,
        "wall_time_sec" => 0.0,
        "note" => "missing MINLPTests symbol: " * symbol_name_str,
    )
end

function run_case(
    optimizer,
    problem_id::AbstractString,
    symbol_name::AbstractString,
    minlptests,
)
    problem_id_str = String(problem_id)
    symbol_name_str = String(symbol_name)
    f = getfield(minlptests, Symbol(symbol_name_str))
    ts = Test.DefaultTestSet(problem_id_str)
    Test.push_testset(ts)
    err_text = nothing
    wall_time = @elapsed begin
        try
            Base.invokelatest(
                f,
                optimizer,
                minlptests.OPT_TOL,
                minlptests.PRIMAL_TOL,
                minlptests.DUAL_TOL,
                minlptests.TERMINATION_TARGET_GLOBAL,
                minlptests.PRIMAL_TARGET_GLOBAL,
            )
        catch err
            err_text = sprint(showerror, err, catch_backtrace())
        end
    end
    Test.pop_testset()

    counts = Test.get_test_counts(ts)
    outcome = (err_text === nothing && counts.fails == 0 && counts.errors == 0 && counts.broken == 0) ? "pass" : "fail"
    note = err_text === nothing ? "" : split(err_text, '\n')[1]
    return Dict(
        "problem_id" => problem_id_str,
        "symbol" => symbol_name_str,
        "outcome" => outcome,
        "passes" => counts.passes,
        "fails" => counts.fails,
        "errors" => counts.errors,
        "broken" => counts.broken,
        "wall_time_sec" => wall_time,
        "note" => note,
    )
end

function main()
    if length(ARGS) != 4
        error("usage: alpine_minlptests_status.jl REQUEST.tsv OUTPUT.jsonl MINLPTESTS_PATH TIME_LIMIT_SEC")
    end

    request_path = ARGS[1]
    output_path = ARGS[2]
    minlptests_path = ARGS[3]
    per_instance_time_limit = parse(Float64, ARGS[4])
    mip_backend = get(ENV, "ALPINE_MIP_SOLVER", "highs")

    push!(LOAD_PATH, minlptests_path)
    Base.eval(Main, :(using MINLPTests))
    minlptests = Base.invokelatest(() -> getfield(Main, :MINLPTests))

    optimizer = build_optimizer(per_instance_time_limit, mip_backend)
    cached_records = Dict{String, Dict{String, Any}}()

    open(output_path, "w") do io
        for line in eachline(request_path)
            isempty(strip(line)) && continue
            fields = split(line, '\t')
            if length(fields) != 3
                error("expected 3 tab-separated fields per line in request file")
            end
            problem_id, category, symbol_name = fields
            canonical_symbol_name = resolve_symbol_name(symbol_name, minlptests)

            if canonical_symbol_name === nothing
                record = missing_symbol_record(problem_id, symbol_name)
            else
                cached = get!(cached_records, canonical_symbol_name) do
                    run_case(optimizer, problem_id, canonical_symbol_name, minlptests)
                end
                record = copy(cached)
                record["problem_id"] = String(problem_id)
                record["symbol"] = String(symbol_name)
                if canonical_symbol_name != symbol_name
                    record["canonical_symbol"] = canonical_symbol_name
                end
            end

            record["category"] = category
            record["mip_solver"] = mip_backend
            write_record(io, record)
        end
    end
end

main()
