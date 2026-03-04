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

### execution MLP

``` 
(.venv312) yohan@neon:~/Documents/TP/CSC8608$ python3 TP4/src/train.py --config TP4/configs/baseline_mlp.yaml --model mlp
device: cuda
model: mlp
epochs: 200
epoch=001 loss=1.9512 train_acc=0.3429 val_acc=0.3580 test_acc=0.3500 train_f1=0.2493 val_f1=0.1329 test_f1=0.1357 epoch_time_s=0.1358
...
epoch=200 loss=0.0092 train_acc=1.0000 val_acc=0.5480 test_acc=0.5790 train_f1=1.0000 val_f1=0.5361 test_f1=0.5651 epoch_time_s=0.0004
total_train_time_s=0.2404
train_loop_time=0.6006
```

### execution GCN

``` 
(.venv312) yohan@neon:~/Documents/TP/CSC8608$ python3 TP4/src/train.py --config TP4/configs/gcn.yaml --model gcn
device: cuda
model: gcn
epochs: 200
epoch=001 loss=1.9490 train_acc=0.8786 val_acc=0.5440 test_acc=0.5530 train_f1=0.8740 val_f1=0.5529 test_f1=0.5657 epoch_time_s=0.1618
...
epoch=200 loss=0.0086 train_acc=1.0000 val_acc=0.7600 test_acc=0.8040 train_f1=1.0000 val_f1=0.7436 test_f1=0.7940 epoch_time_s=0.0008
total_train_time_s=0.3379
train_loop_time=0.7454
```

### Comparaison MLP vs GCN

Le MLP obtient `0.5790` en `test_acc` et `0.5651` en `test_f1`, contre `0.8040` et `0.7940` pour le GCN.  
Le GCN fait donc nettement mieux sur Cora, avec un temps d’entraînement seulement un peu plus élevé.

| modèle | test_acc | test_f1 | temps |
|---|---:|---:|---:|
| MLP | 0.5790 | 0.5651 | 0.6006 s |
| GCN | 0.8040 | 0.7940 | 0.7454 s |

Sur Cora, le GCN peut dépasser le MLP car il exploite non seulement les features des nœuds, mais aussi le signal du graphe via le voisinage.  
Ce signal relationnel est utile quand des noeuds connectés appartiennent souvent à des classes proches, ce qui correspond à une forme d’homophilie.  
Les voisins apportent donc de l’information supplémentaire que le MLP ignore complètement.  
Le GCN réalise ainsi un lissage local des représentations, ce qui aide à mieux classer des nœuds ambigus.  
Ici, ce mécanisme semble très bénéfique, car l’écart avec le MLP est large sur `test_acc` et `test_f1`.  
Mais si les features seules étaient déjà très discriminantes, le gain du graphe pourrait être beaucoup plus faible.  
Un lissage trop fort pourrait aussi devenir négatif si des voisins de classes différentes se mélangent trop.  
Dans ce test le graphe de Cora apporte clairement une information utile que le MLP ne capture pas.

## Exercice 4 : Modèle principal : GraphSAGE + neighbor sampling (mini-batch)

### execution GraphSAGE

``` 
(.venv312) yohan@neon:~/Documents/TP/CSC8608$ python3 TP4/src/train.py --config TP4/configs/sage_sampling.yaml --model sage
device: cuda
model: sage
epochs: 200
epoch=001 loss=1.9474 train_acc=0.9714 val_acc=0.6820 test_acc=0.7060 train_f1=0.9710 val_f1=0.6387 test_f1=0.6686 epoch_time_s=0.1184
...
epoch=200 loss=0.0036 train_acc=1.0000 val_acc=0.7800 test_acc=0.8070 train_f1=1.0000 val_f1=0.7727 test_f1=0.8009 epoch_time_s=0.0013
total_train_time_s=0.4085
train_loop_time=0.8553
```

### Comparaison MLP, GCN, GraphSAGE

Le MLP est nettement en dessous des deux modèles de graphe sur Cora.  
Le GCN et GraphSAGE obtiennent des performances très proches, avec un léger avantage final pour GraphSAGE en `test_f1`.  
En contrepartie, GraphSAGE est aussi le plus lent des trois dans cette expérience.  


| modèle | test_acc | test_f1 | temps |
|---|---:|---:|---:|
| MLP | 0.5790 | 0.5651 | 0.6006 s |
| GCN | 0.8040 | 0.7940 | 0.7454 s |
| GraphSAGE | 0.8070 | 0.8009 | 0.8553 s |

Le neighbor sampling accélère l’entraînement parce qu’on ne traite pas tout le graphe à chaque itération. On prend un mini-batch de nœuds, puis seulement un nombre limité de voisins à chaque couche selon le fanout. Cela réduit le coût mémoire et calcul, donc c’est plus scalable sur de grands graphes. Mais on ne voit qu’une partie du voisinage à chaque passage, donc le gradient est plus bruité Le modèle apprend avec une information moins complète qu’en full-batch. Si le fanout est trop petit, on peut manquer des voisins utiles et perdre en performance. C’est encore plus visible avec des hubs, car ces nœuds ont beaucoup de voisins potentiellement importants. Si on augmente le fanout, on récupère un meilleur signal, mais le coût de calcul augmente aussi. Il faut aussi compter le coût CPU du sampling, qui n’est pas gratuit (dans notre cas, Cora est petit, donc le GCN full-batch coûte déjà très peu et l’overhead du sampling/loader peut au final rendre GraphSAGE un peu plus lent). Le compromis est simple : moins de voisins = plus rapide mais plus de variance, plus de voisins = plus stable mais plus coûteux.

## Exercice 5 : Benchmarks ingénieur : temps d’entraînement et latence d’inférence (CPU/GPU)

## Exercice 6 : Synthèse finale : comparaison, compromis, et recommandations ingénieur
