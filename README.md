# Équilibrage de la production intermittente et de la consommation dans les réseaux électriques par Jeux à Champs Moyens

Ce répertoire expose les principaux résultats de mon stage de M1 (Physique théorique) effectué au LPTMS. Ce stage de recherche de 3 mois a été encadré par [Denis Ullmo](https://www.lptms.universite-paris-saclay.fr/membres/ullmo/).

**[Lire le rapport complet (PDF)](./Rapport_de_Stage_M1.pdf)**

## Résumé

Étude de l'équilibre entre production renouvelable intermittente et consommation d’énergie dans les réseaux électriques. Nous utilisons une approche de théorie des jeux à champ moyen (Mean Field Games - MFG) pour décrire le comportement des consommateurs et voir comment il influe sur le réseau.

Une première partie présente l’état de l’art des modèles de réseaux et du cadre théorique des MFG. Après un aperçu de la physique des réseaux électriques, nous considérons un premier modèle qui donne une bonne intuition des résultats attendus dans les cas les plus complexes. La dernière partie explique comment implémenter un modèle plus raffiné et illustre le début de cette implémentation numérique.

**Mots clefs :** Jeux à champs moyens, Réseaux électriques, Systèmes complexes.

## Structure du projet

* **[Rapport_de_Stage_M1.pdf](./Rapport_de_Stage_M1.pdf)** : Le document final du rapport.
* **src/Python/** : Contient les scripts générant les figures 7, 8, 9, 10 et 11 (Analyse et visualisation).
* **src/Julia/** : Contient les scripts générant les figures 2, 3 et 4 (Diagrammes de bifurcation et simulations temporelles haute performance).


## Installation et Utilisation

Ce projet combine Python pour l'analyse de données et Julia pour les simulations numériques lourdes.

### Partie Python

Les dépendances sont listées dans le fichier `requirements.txt`.

1.  Ouvrez un terminal et placez-vous dans le dossier Python :
    ```bash
    cd src/Python
    ```
2.  Installez les bibliothèques nécessaires :
    ```bash
    pip install -r requirements.txt
    ```
3.  Lancez un script :
    ```bash
    python code_figures_7_8.py
    ```

### Partie Julia

L'environnement est géré par les fichiers `Project.toml` et `Manifest.toml` pour garantir la reproductibilité scientifique.

1.  Placez-vous dans le dossier Julia :
    ```bash
    cd src/Julia
    ```
2.  Lancez Julia :
    ```bash
    julia
    ```
3.  Dans le terminal Julia, installez les dépendances automatiquement :
    ```julia
    import Pkg
    Pkg.activate(".")
    Pkg.instantiate()
    ```
4.  Exécutez un script (ou utilisez l'extension VS Code) :
    ```julia
    include("code_figure_2.jl")
    ```
