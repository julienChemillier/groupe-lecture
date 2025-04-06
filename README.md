# Groupe Lecture ENS Lyon L3

Travaux effectués durant le groupe de lecture du S2 à l'ENS de Lyon en L3 avec Aurélien Garivier.
Ma partie s'appuie sur l'étude du papier de recherche des [GAN](https://arxiv.org/pdf/1406.2661) de 2014.

## Travail session n°3
### Etude de la génération de nouvelles images par le GAN
La question est la suivante: un GAN génère t-il réellement de nouvelles images ?  
Pour répondre à cette question, on va étudier des GAN qui ne s'entrainent qu'avec très peu de données.  
Dans le cas à 2 images, on peut parfois voir apparaître une fusion des visages:
![2 Images issues de la BDD](/travail_session3/2_images_donnees.png)
![2 Images générées par le GAN](/travail_session3/2_images_generees.png)

Dans le cas à 3 images, on peut également voir apparaître une fusion des visages:

![3 Images issues de la BDD](/travail_session3/3_images_donnees.png)
![3 Images générées par le GAN](/travail_session3/3_images_generees.png)

Remarques:
- Il est rare que le GAN produise constamment la même image. Cela provient probablement du fait qu'une telle situation engendrerait forcément une loss plus élevée et donc une modification de comportement.
- Les lunettes semblent être un élément très courant à se transmettre d'un visage à l'autre



