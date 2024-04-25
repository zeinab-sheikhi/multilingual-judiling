include("./model/seq_algos.jl")
include("./data/dataloader.jl")
include("./model/evaluation.jl")


function run_matrices(file_name; data_path="data/raw", title="", split_size=0.1)
    
    data = read_csv(data_path, file_name)
    data = split_shuffle(data, split_size=split_size)
    folded_data = fold_data(data, num_folds=5)
    result = cross_validation(folded_data)

    println()
    println(title)
    for (key, value) in result
        println("$key: $value")
    end
    println()    
end


function run_learn_algo(file_name; data_path="data/raw", title="", split_size=0.1)
    
    data = read_csv(data_path, file_name)
    train, val = train_test_split(data, split_size=split_size)
    res = get_result(train, val)
    
    res_learn_train, gpi_learn_train = learn_path(
        train,
        val,
        res["C_train"],
        res["S_train"],
        res["F"],
        res["C_hat_train"],
        res["A"],
        res["i2f"],
        res["f2i"], 
        gold_ind=res["gold_ind_train"],
        S_hat_val=res["S_hat_train"], 
        target_col=:Inflection,
        max_t=JudiLing.cal_max_timestep(pol_noun_train, pol_noun_val, :Inflection)
    )  
    
    # acc_learn_train = array_accuracy(res_learn_train, res["gold_ind_train"])

    # println(acc_learn_train)

    println("Learn Path for ", title)
    println("learn_train")
    println("learn_val")
    println()


end


function run_build_algo(file_name; data_path="data/raw", title="", split_size=0.1)
    
    data = read_csv(data_path, file_name)
    train, val = train_test_split(data, split_size=split_size)
    res = get_result(train, val)
    
    res_build_train = build_path(
        train,
        res["C_train"],
        res["S_train"],
        res["F"],
        res["C_hat_train"],
        res["A"],
        res["i2f"],
        res["gold_ind_train"],
        max_t=JudiLing.cal_max_timestep(train, val, :Inflection))
    
    res_build_val = build_path(
        val,
        res["C_train"],
        res["S_val"],
        res["F"],
        res["C_hat_val"],
        res["A"],
        res["i2f"],
        res["gold_ind_train"],
        max_t=JudiLing.cal_max_timestep(train, val, :Inflection))
    
    acc_build_train = array_accuracy(res_build_train, res["gold_ind_train"])
    acc_build_val = array_accuracy(res_build_val, res["gold_ind_val"])
    
    write_to_csv(res_build_val, val, res["cue_train"], res["cue_train"], "res")
#     JudiLing.write2csv(
#     res_build_train,
#     train,
#     res["cue_train"],
#     res["cue_train"],
#     "polish_noun_build_res.csv",
#     grams = 3,
#     tokenized = false,
#     sep_token = nothing,
#     start_end_token = "#",
#     output_sep_token = "",
#     path_sep_token = ":",
#     target_col = :Inflection,
#     root_dir = @__DIR__,
#     output_dir = "data/processed"
# )

    println()
    println("Build Path for ", title)
    println("build_train: ", acc_build_train)
    println("build_val: ", acc_build_val)
    println()

end
