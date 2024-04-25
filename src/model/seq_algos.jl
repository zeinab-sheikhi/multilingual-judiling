function learn_path(
    data::DataFrame, 
    data_val::DataFrame,
    C_train::Union{JudiLing.SparseMatrixCSC, Matrix},
    S_val::Union{JudiLing.SparseMatrixCSC, Matrix},
    F_train::Union{JudiLing.SparseMatrixCSC, Matrix},
    C_hat_val::Union{JudiLing.SparseMatrixCSC, Matrix},
    A::JudiLing.SparseMatrixCSC,
    i2f::Dict,
    f2i::Dict;
    gold_ind::Union{Nothing, Vector}=nothing,
    S_hat_val::Union{Nothing, Matrix}=nothing,
    max_t::Int64=15, 
    target_col::Union{String, Symbol}=:Inflection,
    max_can::Int64=10,
    threshold::Float64=0.05,
    tokenized::Bool=false,
    grams::Int64=3,
    keep_sep::Bool=false,
    verbose::Bool=true, 
    check_gold_path::Bool=true,
    sep_token::Union{Nothing, String, Char}="_",
    issparse::Symbol=:dense,
    )
        return JudiLing.learn_paths(
            data,
            data_val,
            C_train,
            S_val,
            F_train,
            C_hat_val,
            A,
            i2f,
            f2i,
            gold_ind=gold_ind,
            Shat_val=S_hat_val,
            max_t=max_t,
            max_can=max_can,
            grams=grams,
            threshold=threshold,
            tokenized=tokenized,
            keep_sep=keep_sep,
            target_col=target_col,
            verbose=verbose,
            check_gold_path=check_gold_path,
            issparse=issparse, 
            sep_token=sep_token
        )   
end


function build_path(
    data::DataFrame, 
    C_train::Union{JudiLing.SparseMatrixCSC, Matrix},
    S_val::Union{JudiLing.SparseMatrixCSC, Matrix},
    F_train::Union{JudiLing.SparseMatrixCSC, Matrix},
    C_hat_val::Union{JudiLing.SparseMatrixCSC, Matrix},
    A::JudiLing.SparseMatrixCSC,
    i2f::Dict,
    C_train_ind::Array;
    max_t::Int64=15,
    n_neighbors::Int64=15, 
    verbose::Bool=true)
    
    return JudiLing.build_paths(
        data,
        C_train,
        S_val,
        F_train,
        C_hat_val,
        A,
        i2f,
        C_train_ind,
        max_t=max_t,
        n_neighbors=n_neighbors,
        verbose=verbose
    )
end
