# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 14:45:07 2025

@author: sixti
"""

import unidecode
import os
import csv
from pathlib import Path
print("Racine du projet :", os.getcwd())
base_dir = Path.cwd()
os.chdir("C:/Users/sixti/Documents/3A/Stat M2/Projet CV")

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
import re
import seaborn as sns
import pdfplumber
from markdown_strings import header, bold, italics, code_block
import markdown

### Importer le jeu de données test

cv=["1CV-barman.pdf","1CV-vendeur.pdf","1CV-boucher.pdf","1cv-patissier.pdf"]
of=["1Offre-barman.pdf","1Offre-vendeur.pdf","1Offre-boucher.pdf","1Offre-patissier.pdf"]

### Préparation des données

# Extaction propre en 1 bloc

def clean_text(txt):
    replacements = {
        "*": "",
        "’": "'",
        "‘": "'",
        "\t": "",
        "\n": " ",
        "\r": "",
    }
    for old, new in replacements.items():
        txt = txt.replace(old, new)
    return txt.strip()

def text (path):
    text=[]
    for i in range (len(path)):
        readercvi = PdfReader(path[i])
        text_pagescvi = [page.extract_text() for page in readercvi.pages]
        full_textcvi = "\n".join(text_pagescvi)
        full_textcvi = clean_text(full_textcvi)
        clean_textcvi = unidecode.unidecode(full_textcvi)
        text=text+[clean_textcvi]
    return text

of_bloc=text(of)
cv_bloc=text(cv)

# Extraction en trois blocs

def segmenter_texte_par_mots_cles(texte):
    """
    Fonction qui range le texte dans les cases selon les mots-clés.
    """
    sections = {
        "Experience": [],
        "Formation": [],
        "Competences": [],
        "Autre": []
    }
    
    # Mots-clés (en minuscules)
    keywords = {
        "Experience": ["experience", "expériences", "expérince", "parcours", "postes", "emploi", "expériences professionelles"],
        "Formation": ["formation", "education", "diplomes", "études", "cursus"],
        "Competences": ["competences", "compétence", "compétences", "hard skills", "outils", "langues", "logiciels", "atouts", "compétences techniques"]
    }
    
    current_section = "Autre" # Section par défaut
    
    lignes = texte.split('\n')
    for ligne in lignes:
        ligne_clean = ligne.strip()
        if not ligne_clean: continue
    # Détection du changement de section (Si la ligne est un titre court)
        ligne_lower = ligne_clean.lower()
        if len(ligne_clean) < 40: # Un titre est rarement long
            for section, triggers in keywords.items():
                if any(trigger in ligne_lower for trigger in triggers):
                    current_section = section
                    break
        
        # On ajoute la ligne à la section en cours
        sections[current_section].append(ligne_clean)
        
    return sections

def extraire_cv_colonnes(pdf_path, ratio_colonne=0.35):

   
    """
    Lit un CV en séparant la colonne de gauche (souvent 1/3 de la page)
    de la colonne de droite.
    ratio_colonne=0.35 signifie que la colonne gauche prend 35% de la largeur.
    """
    contenu_final = {
        "Experience": [],
        "Formation": [],
        "Competences": [],
        "Autre": []
    }

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                width = page.width
                height = page.height
                
                # --- DÉCOUPAGE GÉOGRAPHIQUE ---
                
                # 1. Définir la zone GAUCHE (0 à 35% de la largeur)
                # Bounding box: (x0, top, x1, bottom)
                bbox_gauche = (0, 0, width * ratio_colonne, height)
                zone_gauche = page.crop(bbox_gauche)
                texte_gauche = zone_gauche.extract_text() or ""
               
                # 2. Définir la zone DROITE (35% à 100% de la largeur)
                bbox_droite = (width * ratio_colonne, 0, width, height)
                zone_droite = page.crop(bbox_droite)
                texte_droite = zone_droite.extract_text() or ""
               
                # --- STRUCTURATION ---
               
                # On analyse la colonne gauche
                dict_gauche = segmenter_texte_par_mots_cles(texte_gauche)
               
                # On analyse la colonne droite
                dict_droite = segmenter_texte_par_mots_cles(texte_droite)
               
                # --- FUSION DES RÉSULTATS ---
                for key in contenu_final.keys():
                   # On ajoute ce qu'on a trouvé à gauche
                   contenu_final[key].extend(dict_gauche[key])
                   # On ajoute ce qu'on a trouvé à droite
                   contenu_final[key].extend(dict_droite[key])

        # Nettoyage final (convertir les listes en texte propre)
        return {k: " ".join(v) for k, v in contenu_final.items()}

    except Exception as e:
        print(f"Erreur avec pdfplumber sur {pdf_path}: {e}")
        return None

comp=[]
exp=[]
form=[]
for i in range(len(cv)):
    resultat=extraire_cv_colonnes(cv[i-1], ratio_colonne=0.30) 
    comp=comp+[resultat["Competences"]]
    exp=exp+[resultat["Experience"]]
    form=form+[resultat["Formation"]]
    
### Vectorisation

# Model SentenceTransformer

model_st = SentenceTransformer('all-mpnet-base-v2')

# Model Universal Sentence Encoder

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model_use = hub.load(module_url)

# Vectorisation 1 bloc

cv_bloc_st = np.array([model_st.encode(t) for t in cv_bloc])
of_bloc_st = np.array([model_st.encode(t) for t in of_bloc])

embeddings_1bloc_st = np.vstack((cv_bloc_st,of_bloc_st))

cv_bloc_use = model_use(cv_bloc) 
of_bloc_use = model_use(of_bloc)

of_bloc_use=of_bloc_use.cpu().numpy()
cv_bloc_use = cv_bloc_use.cpu().numpy()

embeddings_1bloc_use = np.vstack((cv_bloc_use,of_bloc_use))


# Vectorisation en 3 blocs 

emb_comp_st = np.array([model_st.encode(t) for t in comp])
emb_exp_st = np.array([model_st.encode(t) for t in exp])
emb_form_st = np.array([model_st.encode(t) for t in form])

emb_comp_use = model_use(comp) 
emb_exp_use = model_use(exp) 
emb_form_use = model_use(form) 

### Calcul des distances

# Avec 1 bloc

similarities_bloc_st = cosine_similarity(cv_bloc_st, of_bloc_st)
distance_bloc_st = 1 - similarities_bloc_st

def plot_matrice_distance(distance,cv,of):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        distance,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        square=True,
        cbar_kws={'label': 'Distance'},
        linewidths=0.5,
        linecolor='grey'
    )
    plt.xticks(ticks=np.arange(len(of)) + 0.5, labels=of, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(of)) + 0.5, labels=cv, rotation=0)

    plt.title("Matrice de distances obtenue avec", fontsize=16)
    plt.xlabel("CV")
    plt.ylabel("Offre")
    plt.tight_layout()
    plt.show()

plot_matrice_distance(distance_bloc_st,cv,of)

similarities_bloc_use = cosine_similarity(cv_bloc_use, of_bloc_use)
distance_bloc_use = 1 - similarities_bloc_use

plot_matrice_distance(distance_bloc_use,cv,of)

# Avec 3 blocs

distance_comp_st = 1 - cosine_similarity(emb_comp_st, of_bloc_st)
distance_exp_st = 1 - cosine_similarity(emb_exp_st, of_bloc_st)
distance_form_st = 1 - cosine_similarity(emb_form_st, of_bloc_st)

distance_3blocs_st = np.zeros((4, 4))
for i in range(len(cv)):
    for j in range(len(cv)):
        distance_3blocs_st[i,j] = distance_comp_st[i,j]+distance_exp_st[i,j]+distance_form_st[i,j]

plot_matrice_distance(distance_3blocs_st,cv,of)

distance_comp_use = 1 - cosine_similarity(emb_comp_use, of_bloc_use)
distance_exp_use = 1 - cosine_similarity(emb_exp_use, of_bloc_use)
distance_form_use = 1 - cosine_similarity(emb_form_use, of_bloc_use)

distance_3blocs_use = np.zeros((4, 4))
for i in range(len(cv)):
    for j in range(len(cv)):
        distance_3blocs_use[i,j] = distance_comp_use[i,j]+distance_exp_use[i,j]+distance_form_use[i,j]

plot_matrice_distance(distance_3blocs_use,cv,of)



