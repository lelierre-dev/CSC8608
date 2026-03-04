# CI : Graph Neural Networks

Yohan Delière
lien github : https://github.com/lelierre-dev/CSC8608
en local


## Exercice 1 : Initialisation du TP et smoke test PyG (Cora)

![alt text](img/image.png)

#### smoke test 

![alt text](img/image-1.png)

## Exercice 2 : Baseline tabulaire : MLP (features seules) + entraînement et métriques

On calcule séparément les métriques sur train_mask, val_mask et test_mask pour suivre le modèle proprement à chaque étape. Le score sur train_mask montre s’il apprend bien sur les données vues pendant l’entraînement. Le score sur val_mask sert à comparer les réglages sans biaiser l’évaluation finale. Le score sur test_mask est gardé à la fin pour estimer le vrai niveau du modèle sur des données jamais utilisées. Cette séparation évite de se tromper sur les performances réelles.

![alt text](img/image-2.png)

![alt text](img/image-3.png)



## Exercice 3 : Baseline GNN : GCN (full-batch) + comparaison perf/temps

## Exercice 4 : Modèle principal : GraphSAGE + neighbor sampling (mini-batch)

## Exercice 5 : Benchmarks ingénieur : temps d’entraînement et latence d’inférence (CPU/GPU)

## Exercice 6 : Synthèse finale : comparaison, compromis, et recommandations ingénieur