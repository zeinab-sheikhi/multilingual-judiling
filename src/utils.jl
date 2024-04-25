using Statistics


function str2list(embedding)
    return parse.(Float64, split(embedding))
end


function round_mean(input_list)
    return round(mean(input_list), digits=4)
end
