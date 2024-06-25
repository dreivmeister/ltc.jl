using Flux


function add_synapse(adj_matrix, src, dest, polarity)
    adj_matrix[src, dest] = polarity
end

struct FullyConnected
    units::Int
    input_dim::Int
    output_dim::Int
    adjacency_matrix::Matrix{Int32}
    sensory_adjacency_matrix::Matrix{Int32}
end

function FullyConnected(units::Int64, input_dim::Int64, output_dim=nothing)
    if isnothing(output_dim)
        output_dim = units
    end

    adj_matrix = Int32.(zeros(units, units))
    for src in 1:units
        for dest in 1:units
            polarity = rand([-1, 1, 1])
            add_synapse(adj_matrix, src, dest, polarity)
        end
    end

    sens_adj_matrix = Int32.(zeros(input_dim, units))
    for src in 1:input_dim
        for dest in 1:units
            polarity = rand([-1, 1, 1])
            add_synapse(sens_adj_matrix, src, dest, polarity)
        end
    end

    return FullyConnected(units, input_dim, output_dim, adj_matrix, sens_adj_matrix)
end

function _sigmoid(v_pre, mu, sigma, t)
    v_pre = Flux.unsqueeze(v_pre, ndims(v_pre)+1)
    mu = Flux.unsqueeze(mu, 1)
    println("vpre", size(v_pre), " ", "mu", size(mu))
    mues = v_pre .- mu
    println("mues", size(mues))
    sigma = Flux.unsqueeze(sigma, 1)
    x = sigma .* mues
    println("x", size(x))
    return Flux.sigmoid(x)
end

function ode_solve(a, input, h_state, ts)
    #dh_state/dt = f(h_state, input, t)
    input = input'
    print("input", size(input), "state", size(h_state))
    ode_unfolds = 6
    v_pre = h_state

    # (2, 5) * (3, 2, 5)
    sensory_w_activation = Flux.unsqueeze(a.sensory_w, dims=1) .* _sigmoid(input, a.sensory_mu, a.sensory_sigma, false)
    print("senswact", size(sensory_w_activation))
    # (3, 2, 5) *= (2, 5)
    sensory_w_activation .*= Flux.unsqueeze(a.sensory_spars_mask, dims=1)
    print("senswact", size(sensory_w_activation))

    # (3, 2, 5) .* (2, 5)
    sensory_rev_activation = sensory_w_activation .* Flux.unsqueeze(a.sensory_erev, dims=1)
    print("sensrevact", size(sensory_rev_activation))

    # (3, 2, 5) -> (3, 5)
    w_numerator_sensory = dropdims(sum(sensory_rev_activation, dims=2), dims=2)
    println("wnumsens", size(w_numerator_sensory))
    w_denominator_sensory = dropdims(sum(sensory_w_activation, dims=2), dims=2)
    println("wdenomsens", size(w_denominator_sensory))

    cm_t = a.cm ./ (ts / ode_unfolds)

    w_param = copy(a.w)
    for t in 1:ode_unfolds
        # (5, 5) .* (3, 5, 5)
        w_activation = Flux.unsqueeze(w_param, dims=1) .* _sigmoid(v_pre, a.mu, a.sigma, false)
        # (3, 5, 5) *= (5,5)
        w_activation .*= Flux.unsqueeze(a.spars_mask, dims=1)
        rev_activation = w_activation .* Flux.unsqueeze(a.erev, dims=1)
        w_numerator = dropdims(sum(rev_activation, dims=2), dims=2) .+ w_numerator_sensory
        w_denominator = dropdims(sum(w_activation, dims=2), dims=2) .+ w_denominator_sensory

        gleak = a.gleak
        numerator = Flux.unsqueeze(cm_t, dims=1) .* v_pre .+ Flux.unsqueeze(gleak, dims=1) .* Flux.unsqueeze(a.vleak, dims=1) .+ w_numerator
        denominator = Flux.unsqueeze(cm_t, dims=1) .+ Flux.unsqueeze(gleak, dims=1) .+ w_denominator

        v_pre = numerator ./ (denominator .+ 1e-8)
    end
    return v_pre
end


struct LTCCell
    wiring
    in_features
    input_w
    input_b
    output_w
    output_b
    sensory_w
    sensory_mu
    sensory_sigma
    sensory_spars_mask
    sensory_erev
    cm
    w
    mu
    sigma
    spars_mask
    erev
    gleak
    vleak
end

function LTCCell(wiring, in_features)
    return LTCCell(wiring, in_features, 
                   ones(wiring.input_dim), zeros(wiring.input_dim),
                   ones(wiring.output_dim), zeros(wiring.output_dim),
                   randn(wiring.input_dim, wiring.units),
                   randn(wiring.input_dim, wiring.units),
                   randn(wiring.input_dim, wiring.units),
                   abs.(wiring.sensory_adjacency_matrix),
                   copy(wiring.sensory_adjacency_matrix),
                   randn(wiring.units),
                   randn(wiring.units, wiring.units),
                   randn(wiring.units, wiring.units),
                   randn(wiring.units, wiring.units),
                   abs.(wiring.adjacency_matrix),
                   copy(wiring.adjacency_matrix),
                   randn(wiring.units),
                   randn(wiring.units))
end
# makes trainable
Flux.@functor LTCCell
# forward pass
function (a::LTCCell)(input, h_state, ts=1.0)
    # input (C, B), h_state (B, units), ts float
    # map_inputs
    input = a.input_w .* input .+ a.input_b


    next_state = ode_solve(a, input, h_state, ts)

    # map_outputs
    output = copy(next_state)
    if a.wiring.output_dim < a.wiring.units
        output = output[:, 1:a.wiring.output_dim]
    end
    # (5,) * (3, 5) * (5,)
    output = Flux.unsqueeze(a.output_w, dims=1) .* output .+ Flux.unsqueeze(a.output_b, dims=1)

    return output, next_state
end


# struct definition
struct LTC
    input_size
    units
    wiring # for now only fully connected
    rnn_cell
end
# constructor
# TODO: what type is input_dim and is input_size == input_dim??
function LTC(input_size::Int, units::Int)
    wiring = FullyConnected(units, input_size)
    return LTC(input_size, units, wiring, LTCCell(wiring, input_size))
end
# forward pass
function (a::LTC)(input, hx=nothing; return_seq=false)
    #TODO: hx is assumed to be nothing right now!!
    # input is of shape (seq_length, input_size, batch_size)
    # (B, L, C) or (L, B, C) and we have (L, C, B)
    seq_length, input_size, batch_size = size(input)
    h_state = zeros(batch_size, a.units) # (B, U)
    h_out = zeros(batch_size, a.units) # (U,)

    output_sequence = []
    for t in 1:seq_length
        inputs = input[t,:,:] # (C, B)
        ts = 1.0
        h_out, h_state = a.rnn_cell(inputs, h_state, ts)
        if return_seq
            push!(output_sequence, h_out')
        end
    end
    if return_seq
        return stack(output_sequence, dims=1), h_state'
    end
    return h_out', h_state'
end

ltc = LTC(2, 5) # (C, U)

input = rand(10, 2, 3) # (L, C, B) # batch dimension is the last
out, state = ltc(input; return_seq=true) # (L, U, B), (U, B) -> (10, 5, 3), (5, 3)
println(size(out), " ", size(state)) # here also batch dimension is the last

# with return_seq=true: (10, 5, 3), (5, 3)
