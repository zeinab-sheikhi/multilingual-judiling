function matrix_accuracy(SChat::Core.AbstractArray, SC::Core.AbstractArray)
    return JudiLing.eval_SC(SChat, SC)
end


function array_accuracy(res::Array, gold_inds::Array)
    return JudiLing.eval_acc(
    res,
    gold_inds,
    verbose=false
)
end
