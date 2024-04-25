using DataFrames
using JudiLing
using SparseArrays


include("./evaluation.jl")
include("../utils.jl")


function combine_cue_matrix(
    train_data::DataFrame,
    val_data::DataFrame;
    grams::Int64=3, 
    target_col::Union{String, Symbol}=:Inflection,
    tokenized::Bool=false,
    verbose::Bool=false, 
    keep_sep::Bool=false, 
    start_end_token::Union{String, Char}="#", 
    sep_token::Union{Nothing, String, Char}=nothing)
    
    return JudiLing.make_combined_cue_matrix(
        train_data,
        val_data,
        grams=grams,
        target_col=target_col,
        tokenized=tokenized,
        keep_sep=keep_sep, 
        verbose=verbose, 
        start_end_token=start_end_token, 
        sep_token=sep_token
        )
end

function get_C(X::JudiLing.Cue_Matrix_Struct)
    return X.C
end

function get_S(df)
    df.Embeddings = map(str2list, df.Embeddings)
    num_columns = length(df.Embeddings[1])
    matrix = zeros(Float64, nrow(df), num_columns)
    for (i, row) in enumerate(eachrow(df))
        matrix[i, :] = row.Embeddings
    end
    return matrix
end

function transform_matrix(X::Union{SparseMatrixCSC, Matrix}, Y::Union{SparseMatrixCSC, Matrix}) 
    return JudiLing.make_transform_matrix(X, Y)
end

function adjacency_matrix(X::JudiLing.Cue_Matrix_Struct)
    return X.A
end

function get_result(X_train, X_val)
        
    cue_train, cue_val = combine_cue_matrix(X_train, X_val)
    
    C_train = get_C(cue_train)
    C_val = get_C(cue_val)
    S_train = get_S(X_train[:, [:Embeddings]])
    S_val = get_S(X_val[:, [:Embeddings]])
        
    G_train = transform_matrix(S_train, C_train)
    F_train = transform_matrix(C_train, S_train)
        
    C_hat_train = S_train * G_train
    C_hat_val = S_val * G_train
    S_hat_train = C_train * F_train
    S_hat_val = C_val * F_train

    C_train_accuracy = matrix_accuracy(C_hat_train, C_train)
    C_val_accuracy = matrix_accuracy(C_hat_val, C_val)
    S_train_accuracy = matrix_accuracy(S_hat_train, S_train)
    S_val_accuracy = matrix_accuracy(S_hat_val, S_val)
    
    res = Dict(
        "cue_train" => cue_train,
        "cue_val" => cue_val,
        "C_train" => C_train,
        "C_hat_train" => C_hat_train,
        "C_val" => C_val,
        "C_hat_val" => C_hat_val,
        "S_train" => S_train,
        "S_hat_train" => S_hat_train,
        "S_val" => S_val,
        "S_hat_val" => S_hat_val,
        "F" => F_train,
        "G" => G_train,
        "A" => adjacency_matrix(cue_train),
        "f2i" => cue_train.f2i,
        "i2f" => cue_train.i2f,
        "gold_ind_train" => cue_train.gold_ind,
        "gold_ind_val" => cue_val.gold_ind,
        "C_train_accuracy" => C_train_accuracy,
        "C_val_accuracy" => C_val_accuracy,
        "S_train_accuracy" => S_train_accuracy,
        "S_val_accuracy" => S_val_accuracy
    )
    return res
end
