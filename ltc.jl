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

function FullyConnected(units::Int, input_dim::Int, output_dim::Int=nothing)
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


struct LTCCell
    wiring
    in_features
end

function LTCCell(wiring, in_features)
    return LTCCell(wiring, in_features)
end
# makes trainable
Flux.@functor LTCCell
# forward pass
function (a::LTCCell)(x)
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
function LTC(input_size::Int, units::Int, input_dim)
    wiring = FullyConnected(units, input_dim)
    return LTC(input_size, units, wiring, LTCCell(wiring, input_size))
end
# forward pass
function (a::LTC)(input, hx=nothing)
    #TODO: hx is assumed to be nothing right now!!
    # input is of shape (seq_length, input_size, batch_size)
    seq_length, input_size, batch_size = size(input)
    h_state = zeros(batch_size, a.units)

    output_sequence = []
    for t in 1:seq_length
        # line 175...
    end
end