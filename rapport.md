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

# Rapport de projet - Reinforced Mario

## par Arthur Meyniel, Hugo Fouché et Aurelie Chamouleau

<!-- Date + Nom de l'école + logo ESEO -->

<!-- Table des matières -->

## Introduction

Le jeu Super Mario Bros, depuis sa sortie dans les années 1980, a captivé des millions de joueurs à travers le monde. Au-delà de son succès commercial, ce jeu iconique présente un intérêt particulier pour le domaine de l'intelligence artificielle, spécialement dans l'étude de l'apprentissage par renforcement.

### Contexte et Pertinence du Jeu Super Mario Bros pour l'Apprentissage par Renforcement

Super Mario Bros offre un environnement complexe et dynamique, idéal pour tester et développer des algorithmes d'apprentissage par renforcement. Avec ses niveaux variés, ses obstacles imprévisibles et ses objectifs multiples, le jeu pose des défis qui imitent des problèmes réels dans le domaine de l'IA. La capacité d'un agent à apprendre et à s'adapter dans cet environnement peut fournir des insights précieux sur les applications de l'apprentissage par renforcement dans des situations complexes.

### Introduction à l'Apprentissage par Renforcement et le Choix du PPO

L'apprentissage par renforcement est une branche de l'intelligence artificielle où un agent apprend à prendre des décisions en interagissant avec son environnement. L'objectif est de maximiser une certaine notion de récompense cumulative. Parmi les différents algorithmes existants, le Proximal Policy Optimization (PPO) se distingue par sa robustesse et sa capacité à gérer efficacement des environnements à grande échelle, comme Super Mario Bros. Le PPO, reconnu pour sa facilité d'implémentation et sa stabilité d'apprentissage, est donc un choix naturel pour notre projet.

## Explication et Formalisation du Problème

### Super Mario Bros : Un Défi pour l'Intelligence Artificielle

Super Mario Bros, un jeu de plateforme développé par Nintendo, est caractérisé par sa structure de niveaux variés, ses ennemis divers, et ses mécaniques de jeu dynamiques. Chaque niveau présente un ensemble unique de défis, allant de la navigation à travers des plateformes en mouvement à l'évitement d'ennemis, tout en collectant des pièces et en atteignant le drapeau de fin de niveau. Ces éléments rendent Super Mario Bros particulièrement adapté pour tester les capacités d'adaptation et d'apprentissage d'un agent IA.

### Objectifs de l'Agent dans Super Mario Bros

L'objectif principal de notre agent IA, entraîné à l'aide de l'algorithme Proximal Policy Optimization (PPO), est de maximiser son score en complétant les niveaux le plus rapidement possible tout en minimisant les erreurs, telles que tomber dans des pièges ou être touché par des ennemis. Ceci est réalisé en apprenant à naviguer dans les différents niveaux, en prenant des décisions basées sur les états actuels du jeu, représentés par des images traitées. Le code fourni met en œuvre cette approche en utilisant la librairie gym_super_mario_bros, avec des actions simplifiées via SIMPLE_MOVEMENT et un traitement des images pour optimiser la performance de l'agent.

### Hypothèses et Limitations de l'Étude

Notre approche repose sur plusieurs hypothèses clés :

Simplification visuelle : Le traitement des images en noir et blanc et leur réduction en taille visent à simplifier l'espace des caractéristiques sans perdre les informations essentielles pour la prise de décision.
Frame Stacking : L'empilement de plusieurs images consécutives fournit une notion de mouvement et de temporalité, essentielle pour anticiper les actions futures.
Cependant, cette étude comporte des limitations. Premièrement, la simplification des images et la réduction de la complexité du jeu pourraient ne pas capturer toutes les nuances requises pour une généralisation complète à des scénarios plus complexes. Deuxièmement, l'efficacité de l'algorithme PPO dépend fortement du réglage des hyperparamètres, qui peut nécessiter une exploration et un ajustement intensifs. Enfin, notre étude se concentre sur une version spécifique de Super Mario Bros, ce qui pourrait limiter la généralité des conclusions à d'autres versions ou à des jeux similaires.

### Description des données

Les données sont les images du jeu. Elles sont récupérées grâce à la librairie gym-super-mario-bros. Elles sont ensuite traitées pour être utilisées par l'algorithme PPO. Les images sont en couleur et de taille 240x256. Nous avons décidé de les convertir en images en noir et blanc de taille 84x84 afin de diminuer le nombre de dimensions, et donc de paramètres de l'algorithme. De plus, nous appliquons une méthode nommée Frame Stacking, qui consiste à empiler les 4 dernières images pour avoir une notion de mouvement. En effet, une seule image ne permet pas de savoir si Mario se déplace vers la droite ou vers la gauche. Enfin, nous normalisons les images pour que les valeurs soient comprises entre 0 et 1.

### Protocole expérimental

