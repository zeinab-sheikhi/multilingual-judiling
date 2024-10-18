using CSV
using Random
using JudiLing


include("./model/build_matrix.jl")
include("./model/evaluation.jl")


function read_csv(file_path, file_name)
    return DataFrame(CSV.File(joinpath(file_path, file_name)))
end


function write_to_csv(
    res::Array{Array{JudiLing.Result_Path_Info_Struct,1},1}, 
    data::DataFrame, 
    cue_obj_train::JudiLing.Cue_Matrix_Struct, 
    cue_obj_val::JudiLing.Cue_Matrix_Struct, 
    filename::String)
    
    JudiLing.write2csv(
    res,
    data,
    cue_obj_train,
    cue_obj_val,
    filename,
    grams = 3,
    tokenized = false,
    sep_token = nothing,
    start_end_token = "#",
    output_sep_token = "",
    path_sep_token = ":",
    target_col = :Inflection,
    root_dir = @__DIR__,
    output_dir = "data/processed"
)
end


function split_shuffle(df; split_size=0.5)
    shuffled_indices = shuffle(1:nrow(df))
    data_size = round(Int, split_size * nrow(df))
    return df[shuffled_indices[1:data_size], :]
end


function train_test_split(df; split_size=0.1, train_size=0.8, test_size=0.2)
    all_shuffled = shuffle(1:nrow(df))
    data_size = round(Int, split_size * nrow(df))
    train_size = round(Int, train_size * data_size)
    test_size = round(Int, test_size * data_size)
    shuffled_indices = all_shuffled[1:data_size]
    train_indices = shuffled_indices[1:train_size]
    train_df = df[train_indices, :]
    test_indices = shuffled_indices[train_size+1:end]
    test_df = df[test_indices, :]
    return train_df, test_df    
end


function fold_data(X; num_folds=5)
    data = []
    for i in 1:num_folds
        fold_size = size(X, 1) ÷ num_folds
        fold_indices = ((i - 1) * fold_size + 1):(i * fold_size)        
        X_train = vcat(X[1:(fold_indices[1] - 1), :], X[(fold_indices[end] + 1):end, :])
        X_test = X[fold_indices, :]
        push!(data, (X_train, X_test))

    end
    return data
end


function cross_validation(folded_data)
    
    C_train_accuracy_list = []
    C_val_accuracy_list = []
    S_train_accuracy_list = []
    S_val_accuracy_list = []

    for i in eachindex(folded_data)
        
        X_train, X_val = folded_data[i]
        res = get_result(X_train, X_val)
        push!(C_train_accuracy_list, res["C_train_accuracy"])
        push!(C_val_accuracy_list, res["C_val_accuracy"])
        push!(S_train_accuracy_list, res["S_train_accuracy"])
        push!(S_val_accuracy_list, res["S_val_accuracy"])

    end
    
    return Dict(
        "C_train_accuracy" => round_mean(C_train_accuracy_list),
        "C_val_accuracy" => round_mean(C_val_accuracy_list),
        "S_train_accuracy" => round_mean(S_train_accuracy_list),
        "S_val_accuracy" => round_mean(S_val_accuracy_list)
    )
end    
