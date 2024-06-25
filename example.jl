using ltc

input_dimension = 2
num_units = 5 # output_dimension = num_units
time_length = 1
batch_size = 3

ltc = LTC(input_dimension, num_units)

input = rand(time_length, input_dimension, batch_size)
out, state = ltc(input; return_seq=true)
println(size(out), " ", size(state)) # expect: if return_seq==false: (num_units, batch_size) and (num_units, batch_size)
                                     #         else (time_length, num_units, batch_size) and (num_units, batch_size)