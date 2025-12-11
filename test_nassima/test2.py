# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 15:00:35 2025

@author: sixti
"""

import unidecode
import os
from pathlib import Path

print("Racine du projet :", os.getcwd())
base_dir = Path.cwd()

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 1. Fonctions utilitaires
# =========================

def clean_text(txt):
    """Nettoyage simple du texte brut."""
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


def decouper_cv_en_blocs(texte):
    """Découpe un CV en 3 blocs : compétences, expériences, formations."""
    texte = texte.lower()

    blocs = {
        "competences": "",
        "experiences": "",
        "formations": ""
    }

    # Mots-clés possibles pour chaque bloc
    keys = {
        "competences": ["competences", "skills"],
        "experiences": ["experience", "experiences", "projets", "stages"],
        "formations": ["formation", "education", "etudes", "diplome"]
    }

    # Ici découpage simple
    lignes = texte.split(".")
    current_bloc = None

    for ligne in lignes:
        for bloc, mots in keys.items():
            if any(mot in ligne for mot in mots):
                current_bloc = bloc

        if current_bloc:
            blocs[current_bloc] += ligne + " "

    return blocs


def embedding_cv_automatique(pdf_path, model):
    reader = PdfReader(pdf_path)
    text_pages = [page.extract_text() for page in reader.pages]

    texte = "\n".join(text_pages)
    texte = clean_text(texte)
    texte = unidecode.unidecode(texte)

    blocs = decouper_cv_en_blocs(texte)

    emb_comp = model.encode(blocs["competences"])
    emb_exp  = model.encode(blocs["experiences"])
    emb_form = model.encode(blocs["formations"])

    # Moyenne pour garder 768 dimensions
    # Essayer la moyenne pondérée (exemple expériences et compétences ont pas le meme poids)
    emb_final = (emb_comp + emb_exp + emb_form) / 3
    emb_final = normalize(emb_final.reshape(1, -1))[0]

    return emb_final



def embedding_offre_automatique(pdf_path, model):
    """
    Embedding global pour une offre (pas de découpage en blocs).
    """
    reader = PdfReader(pdf_path)
    text_pages = [page.extract_text() for page in reader.pages]

    texte = "\n".join(text_pages)
    texte = clean_text(texte)
    texte = unidecode.unidecode(texte)

    emb = model.encode(texte)
    emb = normalize(emb.reshape(1, -1))[0]

    return emb


# Pour récupérer les textes bruts (pour affichage des labels sur le dernier graphique)
def textcv(path_list):
    cv_text = []
    for fname in path_list:
        reader = PdfReader(fname)
        text_pages = [page.extract_text() for page in reader.pages]
        full_text = "\n".join(text_pages)
        full_text = clean_text(full_text)
        full_text = unidecode.unidecode(full_text)
        cv_text.append(full_text)
    return cv_text


def textof(path_list):
    of_text = []
    for fname in path_list:
        reader = PdfReader(fname)
        text_pages = [page.extract_text() for page in reader.pages]
        full_text = "\n".join(text_pages)
        full_text = clean_text(full_text)
        full_text = unidecode.unidecode(full_text)
        of_text.append(full_text)
    return of_text


# =========================
# 2. Listes des fichiers
# =========================

listecv = [
    "Cvsixtine.pdf",
    "CVnassima.pdf",
    "Cvvalentine.pdf",
    "Cvcamille.pdf",
    "cvchimiste.pdf",
    "cvcommerce.pdf",
    "cvchef.pdf",
    "directeur financier.pdf",
    "designer stylist.pdf",
    "sales & trading intern.pdf"
]

listeof = [
    "Offretest.pdf",
    "Offrebucheron.pdf",
    "Offrecomptable.pdf",
    "Offregardien.pdf",
    "Offremodel.pdf",
    "aide soignant.pdf",
    "ambulancier.pdf",
    "animateur 2D_3D.pdf",
    "choregraphe.pdf",
    "kiné.pdf"
]

# Textes bruts (uniquement pour les labels des derniers graphes)
cv_text = textcv(listecv)
of_text = textof(listeof)


# =========================
# 3. Embeddings CV & Offres
# =========================

model = SentenceTransformer('all-mpnet-base-v2')

# CV : embeddings avec découpage en blocs
cv_embeddings = []
for cv_file in listecv:
    emb = embedding_cv_automatique(cv_file, model)
    cv_embeddings.append(emb)
cv_embeddings = np.array(cv_embeddings)

# Offres : embeddings globaux
of_embeddings = []
for offre_file in listeof:
    emb = embedding_offre_automatique(offre_file, model)
    of_embeddings.append(emb)
of_embeddings = np.array(of_embeddings)

# DataFrames pour PCA
datacv = pd.DataFrame(cv_embeddings)
dataof = pd.DataFrame(of_embeddings)


# =========================
# 4. PCA 1 : CV actifs, offres projetées
# =========================

pca_cvof = PCA(n_components=2)
X_pca = pca_cvof.fit_transform(datacv.values)        # CV = individus actifs
offers_pca = pca_cvof.transform(dataof.values)       # Offres projetées

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], label="CV existants")
for i in range(X_pca.shape[0]):
    plt.text(X_pca[i, 0]+0.01, X_pca[i, 1]+0.01, f'CV {i+1}')
plt.scatter(offers_pca[:, 0], offers_pca[:, 1], label='Offres', color='red')
for j in range(offers_pca.shape[0]):
    plt.text(offers_pca[j, 0]+0.01, offers_pca[j, 1]+0.01, f'Offre {j+1}', color='red')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title('ACP des embeddings de CV (offres en supplémentaires)')
plt.legend()
plt.show()


# =========================
# 5. PCA 2 : Offres actives, CV projetés
# =========================

pca_off = PCA(n_components=2)
X_pca2 = pca_off.fit_transform(dataof.values)        # Offres = individus actifs
offers_pcacv = pca_off.transform(datacv.values)      # CV projetés

plt.figure()
plt.scatter(X_pca2[:, 0], X_pca2[:, 1], label="Offres existantes")
for i in range(X_pca2.shape[0]):
    plt.text(X_pca2[i, 0]+0.01, X_pca2[i, 1]+0.01, f'Offre {i+1}')
plt.scatter(offers_pcacv[:, 0], offers_pcacv[:, 1], label='CV', color='red')
for j in range(offers_pcacv.shape[0]):
    plt.text(offers_pcacv[j, 0]+0.01, offers_pcacv[j, 1]+0.01, f'CV {j+1}', color='red')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("ACP des embeddings d'offres (CV en supplémentaires)")
plt.legend()
plt.show()


# =========================
# 6. PCA 3 : PCA sur tous les embeddings
# =========================

pca_all = PCA(n_components=2)
all_embeddings = np.vstack([of_embeddings, cv_embeddings])

all_pca = pca_all.fit_transform(all_embeddings)
of_pca = all_pca[:len(of_embeddings), :]
cv_pca = all_pca[len(of_embeddings):, :]

plt.figure(figsize=(8, 6))

# Offres
plt.scatter(of_pca[:, 0], of_pca[:, 1], label='Offres', color='red')
for i in range(of_pca.shape[0]):
    plt.text(of_pca[i, 0]+0.01, of_pca[i, 1]+0.01, f'Offre {i+1}', color='red')

# CV
plt.scatter(cv_pca[:, 0], cv_pca[:, 1], label='CV', color='blue')
for i in range(cv_pca.shape[0]):
    plt.text(cv_pca[i, 0]+0.01, cv_pca[i, 1]+0.01, f'CV {i+1}', color='blue')

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Projection PCA commune des CV et Offres")
plt.legend()
plt.show()


# =========================
# 7. Similarités cosinus
# =========================

similarities = cosine_similarity(cv_embeddings, of_embeddings)
# similarities[i, j] = similarité entre CV i et offre j

best_matches = similarities.argmax(axis=1)
for i, j in enumerate(best_matches):
    print(f"{listecv[i]} correspond le mieux à {listeof[j]} (similarité={similarities[i, j]:.2f})")
