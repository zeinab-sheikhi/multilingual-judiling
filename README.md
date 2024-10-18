# Advanced Morphology: Training a Linear Discriminative Learning Model

## Introduction
This project aims to train a Linear Discriminative Learning (LDL) model for morphological analysis of three languages—Polish, German, and Italian. Inspired by the study of Baayen et al. (2019), this work evaluates the model's ability to predict both word forms and semantics. Using the Unimorph dataset for morphological data and FastText embeddings for semantics, the project explores how well the LDL model generalizes across different language families.

## Libraries Used
- **[JudiLing.jl](https://github.com/MegamindHenry/JudiLing.jl)**: A Julia library for training and evaluating LDL models.
- **FastText**: For pretrained word embeddings that provide semantic representations.
- **Unimorph Dataset**: For morphological data on Polish, German, and Italian.

## Results Analysis
- **Sense Prediction**: On the train set, sense prediction achieves over 90% accuracy for all categories in German and for nouns and adjectives in Polish. On the test set, only Polish adjectives and verbs maintain over 80% accuracy, while other results vary between 27% for German nouns and 79% for German adjectives.
- **Form Prediction**: The model performs significantly better in sense prediction than in form prediction. Form prediction accuracy reaches over 50% for all categories in Polish and German nouns but drops to 8% for Italian verbs on the test set.
- **Model Outputs**: The model correctly produces words such as *‘mietowy’* (mint-colored) and *‘dwuspadowy’* (gabled, of a roof). Notably, it outputs the synonym *‘niemale’* for *‘spore’* (meaning quite big), indicating semantic closeness, despite being a different form. Additionally, the model generated a non-existent word *‘*niespodziemale’*, which follows Polish morphological rules. Such examples suggest that the model has captured some aspects of the language’s morphology, even if they are not reflected in the accuracy evaluation.
  
## Challenges
- **Difficulty in Form Prediction**: The model struggles with predicting exact forms compared to sense predictions. The lower accuracy for Italian verbs and the complexity of verb morphology may contribute to these challenges.
- **Generalization Across Languages**: While Polish shows the highest accuracy, German and Italian demonstrate more variability in results, particularly for nouns and verbs, highlighting the differences in morphological complexity.
