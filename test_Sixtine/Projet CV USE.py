# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 14:44:39 2025

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

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
  return model(input)

### récupération DATA 

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

### Fonction qui prend les CV et les Offre et renvoie les embeddings

listecv=["Cvsixtine.pdf","CVnassima.pdf","Cvvalentine.pdf","Cvcamille.pdf","cvchimiste.pdf","cvcommerce.pdf","cvchef.pdf","directeur financier.pdf","designer stylist.pdf","sales & trading intern.pdf"]
listeof=["Offretest.pdf","Offrebucheron.pdf","Offrecomptable.pdf","Offregardien.pdf","Offremodel.pdf","aide soignant.pdf","ambulancier.pdf","animateur 2D_3D.pdf","choregraphe.pdf","kiné.pdf"]

def textcv (path):
    cv_text=[]
    for i in range (len(path)):
        readercvi = PdfReader(path[i])
        text_pagescvi = [page.extract_text() for page in readercvi.pages]
        full_textcvi = "\n".join(text_pagescvi)
        full_textcvi = clean_text(full_textcvi)
        clean_textcvi = unidecode.unidecode(full_textcvi)
        cv_text=cv_text+[clean_textcvi]
    return cv_text

def textof (path):
    of_text=[]
    for i in range (len(path)):
        readerofi = PdfReader(path[i])
        text_pagesofi = [page.extract_text() for page in readerofi.pages]
        full_textofi = "\n".join(text_pagesofi)
        full_textofi = clean_text(full_textofi)
        clean_textofi = unidecode.unidecode(full_textofi)
        of_text=of_text+[clean_textofi]
    return of_text

of_text=textof(listeof)
cv_text=textcv(listecv)

cv_emb = embed(cv_text)
of_emb = embed(of_text)

of_emb=of_emb.cpu().numpy()
cv_emb = cv_emb.cpu().numpy()

pca_cvof = PCA(n_components=2)

of_pca = pca_cvof.fit_transform(of_emb)
cv_pca = pca_cvof.transform(cv_emb)

plt.figure(figsize=(8,6))

plt.scatter(of_pca[:,0], of_pca[:,1], color='red', label='Offres')
for i, txt in enumerate(of_text):
    plt.text(of_pca[i,0]+0.01, of_pca[i,1]+0.01, f'Offre {i+1}', color='red')

plt.scatter(cv_pca[:,0], cv_pca[:,1], color='blue', label='CV')
for i, txt in enumerate(cv_text):
    plt.text(cv_pca[i,0]+0.01, cv_pca[i,1]+0.01, f'CV {i+1}', color='blue')

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Projection PCA des CV et Offres")
plt.legend()
plt.show()

## regarder d'autres méthodes de représentation que ACP UMAP ?

### Calcul de cosine similarity 

similarities = cosine_similarity(cv_emb, of_emb)

best_matches = similarities.argmax(axis=1)
for i, j in enumerate(best_matches):
    simi=similarities[i,j]*100
    print(f"CV {i+1} correspond le mieux à l'offre {j+1} (similarité={simi:.2f}%)")




of_emb=of_emb.cpu().numpy()
cv_emb = cv_emb.cpu().numpy()
embeddings_tensor = np.vstack((cv_emb, of_emb))

with open("embeddingstensor.tsv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")
    for row in embeddings_tensor:
        writer.writerow(row.tolist())

print("Fichier créé : embeddingstensor.tsv")

# ============
# 3. Sauvegarde du fichier metadata.tsv
# Exemple : une liste de labels pour les 10 vecteurs
# ============
labels = [f"vecteur_{i+1}" for i in range(embeddings_tensor.shape[0])]

with open("metadatatensor.tsv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")  # première ligne = nom de colonne obligatoire
    for label in labels:
        writer.writerow([label])

print("Fichier créé : metadatatensor.tsv")








### Similarité
plt.rcParams["font.family"] = "DejaVu Sans"

def plot_similarity(labels, features, rotation):
  corr = np.inner(features, features)
  sns.set(font_scale=1.2)
  g = sns.heatmap(
      corr,
      xticklabels=labels,
      yticklabels=labels,
      vmin=0,
      vmax=1,
      cmap="YlOrRd")
  g.set_xticklabels(labels, rotation=rotation)
  g.set_title("Semantic Textual Similarity")

def run_and_plot(messages_):
  message_embeddings_ = embed(messages_)
  plot_similarity(messages_, message_embeddings_, 90)

run_and_plot(of_text)

