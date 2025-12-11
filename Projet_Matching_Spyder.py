# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 17:48:13 2025

@author: sixti
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import unidecode
import pdfplumber
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

### Configuration des chemins

BASE_DIR = Path.cwd() 
CV_DIR = BASE_DIR / "data" / "cv"
OFFRES_DIR = BASE_DIR / "data" / "offre"

### Préparation des données et extraction des textes

def clean_text(txt):
    if not isinstance(txt, str): return ""
    replacements = {"*": "", "’": "'", "‘": "'", "\t": " ", "\n": " ", "\r": ""}
    for old, new in replacements.items():
        txt = txt.replace(old, new)
    txt = unidecode.unidecode(txt)
    txt = " ".join(txt.split()) 
    return txt.strip()

# Extraction en 1 bloc

def extraire_texte_simple(pdf_path): 
    try:
        reader = PdfReader(pdf_path)
        text = [page.extract_text() for page in reader.pages if page.extract_text()]
        return clean_text(" ".join(text))
    except Exception as e:
        print(f" Erreur Offre {pdf_path.name}: {e}")
        return ""
    
# Extraction en 3 blocs

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
        "Competences": ["competences", "skills", "compétence", "compétences", "hard skills", "outils", "langues", "logiciels", "atouts", "compétences techniques"]
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
    
# Listes pour stocker les données
comp=[]
exp=[]
form=[]    
cv_full_text = [] 
cv_names = []

offre_full_text = []
offre_names = []

# Chargement des CV
for f in sorted(list(CV_DIR.glob("*.pdf"))):
    resultat=extraire_cv_colonnes(f,ratio_colonne=0.35) 
    comp=comp+[resultat["Competences"]]
    exp=exp+[resultat["Experience"]]
    form=form+[resultat["Formation"]]
    if resultat:
        full = extraire_texte_simple(f)
        cv_full_text.append(full)
        cv_names.append(f.name)
        print(f"{f.name} OK")

# Chargement des Offres
for f in sorted(list(OFFRES_DIR.glob("*.pdf"))):
    txt = extraire_texte_simple(f)
    if txt:
        offre_full_text.append(txt)
        offre_names.append(f.name)
        print(f"{f.name} OK")
        
loncv=len(cv_full_text)
lonof=len(offre_names)

### Vectorisation

# Model SentenceTransformer

model_st = SentenceTransformer('all-mpnet-base-v2')

# Model Universal Sentence Encoder

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model_use = hub.load(module_url)

# TEXTE COMPLET (1 BLOC) 

# Sentence Transformer
cv_vec_st = np.array([model_st.encode(t) for t in cv_full_text])
of_vec_st = np.array([model_st.encode(t) for t in offre_full_text])
dist_1bloc_st = 1 - cosine_similarity(cv_vec_st, of_vec_st)

# USE
cv_vec_use = model_use(cv_full_text).numpy()
of_vec_use = model_use(offre_full_text).numpy()
dist_1bloc_use = 1 - cosine_similarity(cv_vec_use, of_vec_use)

# TF-IDF
def vect_tdidf (cv_textes,offre_textes):
    if not cv_textes or not offre_textes:
        print(" ARRÊT : Pas de textes à vectoriser")
    else:
        # Initialiser le vectoriseur
        vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            min_df=1
        )
        try:
            # Vectoriser CV et offres
            X_cv = vectorizer.fit_transform(cv_textes)
            X_offres = vectorizer.transform(offre_textes)
            return[X_cv,X_offres]
        except ValueError as e:
            print(f" ERREUR de vectorisation : {e}")
            
cv_vec_tfidf = vect_tdidf(cv_full_text,offre_full_text)[0]
of_vec_tfidf = vect_tdidf(cv_full_text,offre_full_text)[1]
dist_1bloc_tfidf = 1 - cosine_similarity(cv_vec_tfidf, of_vec_tfidf)

# STRUCTURE (3 BLOCS) 

# On vectorise séparément Expérience, Compétences, Formation

#Sentence Transformer
comp_st = np.array([model_st.encode(t) for t in comp])
exp_st = np.array([model_st.encode(t) for t in exp])
form_st = np.array([model_st.encode(t) for t in form])

#USE
comp_use = model_use(comp).numpy()
exp_use = model_use(exp).numpy()
form_use = model_use(form).numpy()

def calc_dist_3blocs(v_comp, v_exp, v_form, v_offre):
    distance =np.zeros((loncv,lonof))
    d_comp = 1 - cosine_similarity(v_comp, v_offre)
    d_exp = 1 - cosine_similarity(v_exp, v_offre)
    d_form = 1 - cosine_similarity(v_form, v_offre)
    for i in range(loncv):
        for j in range(lonof):
            distance[i,j] = d_comp[i,j]+d_exp[i,j]+d_form[i,j]
    return distance

dist_3blocs_st = calc_dist_3blocs(comp_st, exp_st, form_st, of_vec_st)
dist_3blocs_use = calc_dist_3blocs(comp_use, exp_use, form_use, of_vec_use)

### Visualisation des distances

def plot_heatmap(data, title):
    plt.figure(figsize=(10, 6))
    # cmap="viridis_r" -> Plus c'est foncé, plus la distance est FAIBLE (donc meilleur match)
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="viridis_r", 
        xticklabels=offre_names,
        yticklabels=cv_names
    )
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Affichage des 5 matrices
plot_heatmap(dist_1bloc_st, "Distance - SentenceTransformer (Global)")
plot_heatmap(dist_1bloc_use, "Distance - USE (Global)")
plot_heatmap(dist_1bloc_tfidf, "Distance - TF-IDF (Global)")
plot_heatmap(dist_3blocs_st, "Distance - SentenceTransformer (Structuré 3 blocs)")
plot_heatmap(dist_3blocs_use, "Distance - USE (Structuré 3 blocs)")