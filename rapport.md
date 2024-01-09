<!-- Consignes pour la constitution du rapport et le déroulement de la présentation
Le plan que vous devrez suivre pour votre présentation et le dossier est le suivant :

Explication et formalisation du problème
Description des données
Présentation du travail effectué (protocole expérimental, nettoyage des données, méthodes utilisées, paramétrage des algorithmes, etc)
Résultats obtenus
Discussion et ouverture 
Bibliographie
Précisions :

Les supports de présentation sont à déposer en amont des soutenances. Chaque équipe disposera de 15 min de présentation suivi de 10 min de questions. Attention à la qualité du discours, des supports et à la répartition équitable du temps entre les membres de l'équipe.
Pour le dossier, visez une dizaine de pages. -->

<!-- A tester : pandoc pour passer de Markdown à pdf -->

# Rapport de projet - Reinforced Mario

## par Arthur Meyniel, Hugo Fouché et Aurelie Chamouleau

<!-- Date + Nom de l'école + logo ESEO -->

<!-- Table des matières -->


<!-- Outline finale -->
## Introduction

Le jeu Super Mario Bros, depuis sa sortie dans les années 1980, a captivé des millions de joueurs à travers le monde. Au-delà de son succès commercial, ce jeu iconique présente un intérêt particulier pour le domaine de l'intelligence artificielle, spécialement dans l'étude de l'apprentissage par renforcement.

Super Mario Bros offre un environnement complexe et dynamique, idéal pour tester et développer des algorithmes d'apprentissage par renforcement. Avec ses niveaux variés, ses obstacles imprévisibles et ses objectifs multiples, le jeu pose des défis qui imitent des problèmes réels dans le domaine de l'IA. La capacité d'un agent à apprendre et à s'adapter dans cet environnement peut fournir des insights précieux sur les applications de l'apprentissage par renforcement dans des situations complexes.

## Explication et Formalisation du Problème

## Description des données

Ce projet tire parti des données fournies par `gym-super-mario-bros`, une bibliothèque basée sur la technologie `Gymnasium` développée par OpenAI. L'environnement `SuperMarioBros-v0` est utilisé pour la récupération des données de jeu, qui se présentent sous deux formes principales :

1. Images du Jeu :
    - Chaque image de jeu est capturée avec une résolution de 240x256 pixels, en format couleur RGB.
    - Ces images sont stockées sous forme de tableaux NumPy tridimensionnels de taille 240x256x3. Chaque valeur au sein de ces tableaux représente une intensité de couleur, variant de 0 à 255.

2. Dictionnaire de l'État du Jeu :
    - Les données incluent un dictionnaire exhaustif qui renferme des informations détaillées sur l'état actuel du jeu. Ce dictionnaire comprend :
        - La position de Mario, le score, le temps restant, etc.
        - La récompense obtenue lors de l'étape précédente.
        - Un indicateur booléen signalant si le jeu est terminé.
        - Des informations complémentaires sur le niveau, telles que le nombre de pièces collectées et le temps restant.

### Reward function

La fonction de récompense, élément clé de cet environnement, est conçue autour de l'objectif principal du jeu : maximiser la progression horizontale (à droite) de l'agent, aussi rapidement que possible, tout en évitant la mort. Cette fonction se compose de trois variables distinctes :

1. v - Différence de Position Horizontale : 
    - v = x1 - x0, où x0 et x1 représentent respectivement la position horizontale de Mario avant et après un pas de temps.
    - Une valeur positive (v > 0) indique un mouvement vers la droite, tandis qu'une valeur négative (v < 0) signale un mouvement vers la gauche.

2. c - Différence Temporelle :
    - c = c0 - c1, où c0 et c1 sont les lectures de l'horloge du jeu avant et après un pas de temps.
    - Cette variable sert de pénalité pour dissuader l'agent de rester immobile.

3. d - Pénalité de Mort :
    - Attribue une pénalité significative en cas de décès de l'agent, pour encourager l'évitement de la mort.
    - d = 0 en cas de survie et d = -15 en cas de décès.

La récompense totale, r, est alors calculée comme la somme de ces trois composantes : r = v + c + d. Elle est limitée à l'intervalle [-15, 15].

### Dictionnaire `info`

Le dictionnaire `info`, retourné par la méthode `step`, contient des clés informatives cruciales, telles que :

- `coins` : Le nombre de pièces collectées.
- `flag_get` : Booléen indiquant si Mario a atteint un drapeau ou une hache.
- `life` : Le nombre de vies restantes.
- `score` : Le score cumulatif du jeu.
- `stage` : L'étape actuelle du jeu.
- `status` : Le statut de Mario (petit, grand, avec des boules de feu).
- `time` : Le temps restant sur l'horloge du jeu.
- `world` : Le monde actuel du jeu.
- `x_pos` : La position horizontale de Mario dans l'étape.
- `y_pos` : La position verticale de Mario dans l'étape.

Les données sont mises à jour à chaque frame, offrant ainsi une vue dynamique et détaillée du déroulement du jeu en temps réel.


## Présentation du travail effectué

Le travail est réalisé en plusieurs parties :
- Une étude expérimentale pour comprendre les données récupérées en entrées
- Un choix d'algorithmes d'apprentissage par renforcement -> 2 algorithmes ont été choisis : PPO et DDQN
- Pour chaque algorithme, un nettoyage des données a été réalisé
- Pour chaque algorithme, les impacts des paramètres ont été étudiés
- Pour chaque algorithme, les résultats ont été analysés

### PPO

#### Principe



#### Nettoyage des données et prétraitement

Comme expliqué dans la partie [Description des données](#description-des-données), les données récupérées sont des images de 240x256 pixels en RGB. Cela nous est fourni sous forme d'un tableau numpy de taille 240x256x3, avec des valeurs comprises entre 0 et 255.

Une image du jeu est donc de taille considérable, et il est difficile de faire de l'apprentissage avec une image de cette taille. Il est donc nécessaire de réduire la taille des données en entrée.

Pour cela, nous avons réfléchi à plusieurs solutions :
- Réduire la taille de l'image en la redimensionnant
- Transformer l'image en noir et blanc afin de réduire le nombre de canaux et donc avoir une image de taille 240x256x1. En effet, le jeu est en couleur, mais les couleurs n'ont pas d'importance pour l'apprentissage. De plus, cela permet de réduire la taille des données en entrée par 3.

Nous avons commencé par passer l'image en noir et blanc comme ceci fait la plus grande différence en terme de taille de données. Puis nous avons effectué un prétraitement sur les données en entrée afin de faciliter l'apprentissage. Voici la comparaison entre une image de base et une image en noir et blanc :

![Image de base](./assets/first_frame_color.png) ![Image en noir et blanc](./assets/first_frame_grayscale.png)

De plus, afin d'aider notre modèle, plutôt que de lui envoyer une seule image en entrée, nous lui envoyons les 4 dernières images du jeu. Cela permet de donner du contexte à notre modèle, et de lui permettre de comprendre la vitesse et la direction du personnage et des ennemis du jeu.

Voici un exemple de ce que reçoit notre modèle en entrée (4 premières images du jeu) :

![Frame stack](./assets/frame_stack.png)

##### Sans filtre
##### Avec filtre
#### Paramétrages
#### Résultats

### DDQN
#### Principe
#### Nettoyage des données
#### Paramétrages
#### Résultats

## Comparaison des résultats

## Discussion et ouverture

## Bibliographie

