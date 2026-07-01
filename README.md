# Test technique -- Senior Engineer IA/LLM

## Contexte

Tu rejoins une startup qui developpe un outil pour les etudes notariales. L'equipe souhaite mettre en place un agent conversationnel capable de repondre aux questions des notaires et collaborateurs a partir des pieces d'un dossier de vente.

## Base documentaire

Le dossier `documents/` contient les resultats OCR (format Google Vision API) des pieces de **3 dossiers de vente immobiliere** :

| Dossier | Nombre de pieces |
|---------|-----------------|
| `dossier_1/` | 9 documents |
| `dossier_2/` | 6 documents |
| `dossier_3/` | 6 documents |

Chaque dossier contient :

| Type | Exemples | Format |
|------|----------|--------|
| Pieces d'identite | CNI | JSON (OCR Google Vision) |
| Compromis de vente | Compromis sous seing prive | JSON (OCR Google Vision) |
| Justificatifs de domicile | Facture EDF, avis d'imposition | JSON (OCR Google Vision) |
| DPE | Diagnostic de performance energetique | JSON (OCR Google Vision) |

**~21 fichiers au total.**

Les fichiers JSON suivent le format de reponse de l'API Google Cloud Vision (`fullTextAnnotation` avec `pages`, `blocks`, `words`, `symbols` et scores de `confidence`).

## Objectif

Developper une API permettant :

1. **D'interroger la base documentaire via un LLM** -- un utilisateur doit pouvoir poser des questions en langage naturel et obtenir des reponses basees sur les documents des dossiers.

2. **D'evaluer le cout, latence et la pertinence des reponses** -- un endpoint ou un outil permettant de mesurer et suivre la qualite, le cout et la latence des interactions.

La conception de l'API (endpoints, format des requetes/reponses, architecture) est libre -- c'est au candidat de proposer le design qu'il juge le plus adapte.

## Contraintes techniques

- Python 3.11+
- FastAPI
- LLM : Au choix
- Toute lib RAG/embedding/vectorstore autorisee
- Le code doit tourner avec `docker compose up` ou `make run`
- `README.md` expliquant les choix techniques et d'architecture


## Exemples de questions que l'agent doit savoir traiter

*(Ces exemples illustrent le type de questions attendues -- l'agent doit pouvoir repondre a des questions similaires, pas uniquement a celles-ci)*

- Dans quel dossier monsieur ... est-il vendeur?
- Qui sont les acheteurs et les vendeurs sur le dossier ...?
- Quel est le bien concerne par la transaction de Paris ?
- Les pieces d'identite sont-elles en ordre dans le dossier [1/2/3] ?
- Dans le dossier [1/2/3], Le DPE est-il present et correspond-il au bien ?
- Les justificatifs de domicile sont-ils conformes dans le dossier [1/2/3] ?
- Y a-t-il des incoherences entre les documents du dossier [1/2/3] ?
- Y a-t-il des incohérences dans les informations relatives aux parties du dossier [1/2/3]?
- Fais un resume des pieces manquantes dans chacun des dossier

## Criteres d'evaluation

- L'agent repond de facon pertinente sur la base documentaire fournie
- Le cout, la latence et la pertinence des reponses sont mesurables
- Le code est lisible et bien structure
- Le projet se lance en une commande
- Les choix techniques sont documentes dans le README: y compris une explication des arbitrages réalisés compte tenu des délais impartis, et suggestions pour aller plus loin.

## Duree

Pour discussion 5 jours après réception du test. Envoyer un clone de ce repo avec tes travaux, au moins 24h avant notre discussion.
