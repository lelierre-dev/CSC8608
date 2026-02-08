# CI : Génération d'image

Yohan Delière
lien github : https://github.com/lelierre-dev/CSC8608
en local


## Exercice 1 : Mise en place & smoke test (GPU + Diffusers)

La génération activait à tort le safety trigger NSFW malgré un prompt innocent (montre).
la désactivation du safety checker n'a pas aidé, l’image restait noire.
J'ai ensuite basculé de float16 à float32 sur MPS parce que sur Apple MPS, le calcul en float16 est souvent instable avec Stable Diffusion et peut produire des NaN.

![alt text](img/image-1.png)

![alt text](img/image.png)

## Exercice 2 : Factoriser le chargement du pipeline (text2img/img2img) et exposer les paramètres

```
CONFIG: {'model_id': 'stable-diffusion-v1-5/stable-diffusion-v1-5', 'scheduler': 'EulerA', 'seed': 42, 'steps': 30, 'guidance': 7.5}
```
![alt text](img/image-2.png)

## Exercice 3 : Text2Img : 6 expériences contrôlées (paramètres steps, guidance, scheduler)

![alt text](img/image-3.png)
Run01 – EulerA / 30 steps / guidance 7.5 (baseline)
Rendu “packshot” propre : bouteille bien centrée, fond neutre, reflets cohérents.
Le sceau/couronne doré est lisible et “premium”.
Léger défaut : cadrage un peu serré (bouteille coupée en bas) + pseudo-texte sur l’étiquette.

![alt text](img/image-4.png)
Run02 – EulerA / 15 steps / guidance 7.5
Moins stable : apparition d’une 2ᵉ bouteille dorée (hors-prompt) + composition déséquilibrée.
Détails de l’étiquette plus “mous”, typiques d’un rendu encore “incomplet”.
Impression globale : cohérence en baisse, “drift” plus facile.

![alt text](img/image-5.png)
Run03 – EulerA / 50 steps / guidance 7.5
Plus “fini” : meilleurs reflets sur le verre, bords plus propres, étiquette plus nette.
Look plus premium et contrôlé (moins d’artefacts évidents).
Diminishing returns quand même : c’est surtout du polish (pas un changement radical).

![alt text](img/image-6.png)
Run04 – EulerA / 30 steps / guidance 4.0
Rendu plus doux/naturel, mais moins fidèle au prompt : le sceau/couronne est moins marqué / plus “fondu”.
Détails plus timides (contraste plus faible, moins de “pop” sur le doré).
Bien pour éviter l’effet “trop forcé”, moins moins bien niveau détail.

![alt text](img/image-7.png)
Run05 – EulerA / 30 steps / guidance 12.0
Prompt “sur-insisté” : doré très présent, contraste plus agressif, look très “luxury”.
Néanmoins 2ème bouteille + reflets/artefacts (ex. forme verticale étrange sur la bouteille or).
Guidance trop haute : plus de contrainte, plus de glitches.

![alt text](img/image-8.png)
Run06 – DDIM / 30 steps / guidance 7.5
Changement de mise en scène : fond pas totalement blanc, grande forme doré derrière moins e-commerce.
Style plus "éditorial”, étiquette plus grande et plus structurée.
Globalement plus jolie, mais moins strict sur “fond blanc + produit seul”.

##### Effet des paramètres
###### Steps (15, 30, 50)
plus de steps = plus de netteté (bords, reflets, doré), meilleure cohérence locale.
Trop bas (15) = image “pas finie” + plus de drift (objets en trop).
Au-delà d’un certain point (50) = surtout du polish (gain marginal).

###### Guidance (4, 7.5, 12)
Faible (4) : plus naturel/souple, mais le prompt ressort moins (sceau/couronne moins affirmé).
Moyen (7.5) : bon compromis fidélité et réalisme.
Élevé (12) : éléments “or/premium” sur-accentués, mais risque d’artefacts et d’objets parasites.

###### Scheduler (EulerA vs DDIM)
EulerA : rendu souvent plus “punchy” et détaillé, bon pour packshot net, mais peut aussi amplifier des bizarreries quand la guidance monte.
DDIM : rendu plus “smooth” et parfois plus créatif sur la composition, mais peut s’éloigner du “fond blanc / produit seul”.

## Exercice 4 : Img2Img : 3 expériences contrôlées (strength faible/moyen/élevé)

###### Original :
![alt text](img/image-9.png)

###### force 0,35 :
![alt text](img/image-10.png)

###### force 0,60 :
![alt text](img/image-11.png)

###### force 0,85 :
![alt text](img/image-12.png)


À strength = 0.35, l’identité du produit est largement préservée : forme globale, éléments distinctifs (zips, coutures) et cadrage “dans la main” restent cohérents.
Les changements portent surtout sur des textures (cuir plus lisse/synthétique), des marquages (pseudo-texte), et un éclairage légèrement plus uniforme, tandis que l’arrière-plan sombre reste quasi inchangé.
En e-commerce, ce réglage est relativement utilisable pour rester fidèle au produit, mais non conforme à un packshot “fond blanc” demandé dans le prompt.

À strength = 0.60, certains attributs (teinte brune, présence d’un zip) subsistent, mais l’identité produit commence à dériver fortement : forme beaucoup moins fidèle, détails clés effacés, et main altéré.
Les variations deviennent structurelles : arrière-plan décoratif, rendu plus “studio/3D”, et perte de caractéristiques propres au modèle initial.
En e-commerce, le risque est déjà inacceptable pour une image de cette complexité.

À strength = 0.85, la conservation est minimale (principalement une couleur proche), tandis que la forme, les détails et la scène sont remplacés.
Les textures se simplifient et l’objet devient visuellement différent, rendant la sortie inutilisable pour une fiche produit.

En conclusion, à faible strength, l’objet reste correct mais le fond blanc n’est pas atteint. à fort strength, la scène change davantage mais le modèle altère aussi l’objet.

## Exercice 5 : Mini-produit Streamlit (MVP) : Text2Img + Img2Img avec paramètres

## Exercice 6 :