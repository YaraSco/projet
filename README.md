# projet IFT712

Le présent projet fait la recherche d’hyper-paramètres avec une cross-validations pour identifier la meilleure solution possible pour résoudre le problème de classification.
Il utilise la base de données www.kaggle.com/c/leaf-classification.

## Prérequis:

Ce projet est sous python version au delà de 3.0
Merci d'installer le fichier requirements avec la commande 
```console
$ pip3 install requirements.txt <classifieur> <metrics> <approche_de_resolution_du_probleme>
```
## Run du projet:

La commande à utiliser est 
```console
$ python3 main.py <classifieur> <metrics> <approche_de_resolution_du_probleme>
```
Vous pouvez consulter le fichier main.py pour plus d'information.

## Modèles utilisés:

Nous avons utilsé :
    - La régression logistique
    - Le réseau de neurones
    - AdaBoost
    - RandomForest
    - Le SVM
    - LinearDiscriminantAnalysis
