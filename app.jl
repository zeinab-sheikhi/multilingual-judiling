using Conda
using DataFrames
using JudiLing 


include("src/runner.jl")


# Polish
run_matrices("pol_adj.csv", title="Polish Adjectives", split_size=0.1)
run_matrices("pol_noun.csv", title="Polish Nouns", split_size=0.12)
run_matrices("pol_verb.csv", title="Polish Verbs", split_size=0.6)

# Polish Indicative Verbs
run_matrices("pol_ind_verb.csv", title="Polish Indicative Verbs", split_size=1)

# German
run_matrices("germ_adj.csv", title="German Adjectives", split_size=0.15)
run_matrices("germ_noun.csv", title="German Nouns", split_size=0.05)
run_matrices("germ_verb.csv", title="German Verbs", split_size=0.1)

# German Indicative Verbs
run_matrices("germ_ind_verb.csv", title="German Indicative Verbs", split_size=1)

# Italian
run_matrices("ital_verb.csv", title="Italian Verbs", split_size=0.31)

# Italian Indicative Verbs
run_matrices("it_ind_verb.csv", title="Italian Indicative Verbs", split_size=1)


run_build_algo("pol_adj.csv", title="Polish Adjectives", split_size=0.1)
run_build_algo("pol_noun.csv", title="Polish Nouns", split_size=0.12)
run_build_algo("pol_verb.csv", title="Polish Verbs", split_size=0.6)

run_build_algo("germ_adj.csv", title="German Adjectives", split_size=0.15, output_file_name="german_adj_build_path")
run_build_algo("germ_noun.csv", title="German Nouns", split_size=0.05)
run_build_algo("germ_verb.csv", title="German Verbs", split_size=0.1)

run_build_algo("ital_verb.csv", title="Italian Verbs", split_size=0.31)
