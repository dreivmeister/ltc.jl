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

function _sigmoid(v_pre, mu, sigma)
    #v_pre = Flux.unsqueeze(v_pre, ndims(v_pre)+1)
    mues = v_pre .- mu
    x = sigma .* mues
    return Flux.sigmoid(x)
end

function ode_solve(a, input, h_state, ts)
    #dh_state/dt = f(h_state, input, t)
    ode_unfolds = 6
    v_pre = h_state

    sensory_w_activation = a.sensory_w .* _sigmoid(input, a.sensory_mu, a.sensory_sigma)
    sensory_w_activation .*= a.sensory_spars_mask

    sensory_rev_activation = sensory_w_activation .* a.sensory_erev

    w_numerator_sensory = sum(sensory_rev_activation, dims=2)
    w_denominator_sensory = sum(sensory_w_activation, dims=2)

    cm_t = a.cm ./ (ts / ode_unfolds)

    w_param = copy(a.w)
    for t in 1:ode_unfolds
        w_activation = w_param .* _sigmoid(v_pre, a.mu, a.sigma)
        w_activation .*= a.spars_mask
        rev_activation = w_activation .* a.erev
        w_numerator = sum(rev_activation, dims=2) .+ w_numerator_sensory
        w_denominator = sum(w_activation, dims=2) .+ w_denominator_sensory

        gleak = a.gleak
        numerator = cm_t .* v_pre .+ gleak .* a.vleak .+ w_numerator
        denominator = cm_t .+ gleak .+ w_denominator

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
                   randn(wiring.units,wiring.units),
                   randn(wiring.units,wiring.units),
                   abs.(wiring.adjacency_matrix),
                   copy(wiring.adjacency_matrix),
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
    output = a.output_w .* output .+ a.output_b

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
function (a::LTC)(input, hx=nothing)
    #TODO: hx is assumed to be nothing right now!!
    # input is of shape (seq_length, input_size, batch_size)
    # (B, L, C) or (L, B, C) and we have (L, C, B)
    seq_length, input_size, batch_size = size(input)
    h_state = zeros(batch_size, a.units)
    h_out = zeros(a.units)

    #output_sequence = []
    for t in 1:seq_length
        inputs = input[t,:,:] # (C, B)
        ts = 1.0
        h_out, h_state = a.rnn_cell(inputs, h_state, ts)
        #push!(output_sequence, h_out)
    end
    return h_out, h_state
end

ltc = LTC(2, 5)
input = rand(10, 2, 3)
ltc(input)