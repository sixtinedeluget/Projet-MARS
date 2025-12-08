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
os.chdir("C:/Users/sixti/Documents/3A/Stat M2/Projet CV")

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

import nltk
import spacy
from spacy.cli import download
download("en_core_web_sm")
nltk.download("words")
nltk.download('stopwords')

from pyresparser import ResumeParser
data = ResumeParser('C:/Users/sixti/Documents/3A/Stat M2/Projet CV/CVnassima.pdf').get_extracted_data()

### Automatiser avec une boucle for en simplifiant tous les cv par cv1, cv2 etc
### Pb pas d'anonymisation, extraction en entier, lit de droite à gauche pas par blocs

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

#### Extraction des embeddings CV agro 1

pdf_pathcv1 = "Cvsixtine.pdf"
readercv1 = PdfReader(pdf_pathcv1)
text_pagescv1 = [page.extract_text() for page in readercv1.pages]
full_textcv1 = "\n".join(text_pagescv1)
full_textcv1 = clean_text(full_textcv1)
clean_textcv1 = unidecode.unidecode(full_textcv1)
model = SentenceTransformer('all-MiniLM-L6-v2') # modèle plus rapide 
model = SentenceTransformer('all-mpnet-base-v2') # modèle pré-entrainé plus performant
embeddingscv1 = model.encode(clean_textcv1)
embeddingscv1 = normalize(embeddingscv1.reshape(1, -1))[0]


#### Extraction des embeddings CV agro 2

pdf_pathcv2 = "CVnassima.pdf"
readercv2 = PdfReader(pdf_pathcv2)
text_pagescv2 = [page.extract_text() for page in readercv2.pages]
full_textcv2 = "\n".join(text_pagescv2)
full_textcv2 = clean_text(full_textcv2)
clean_textcv2 = unidecode.unidecode(full_textcv2)
embeddingscv2 = model.encode(clean_textcv2)
embeddingscv2 = normalize(embeddingscv2.reshape(1, -1))[0]

#### Extraction des embeddings CV agroalim 3

pdf_pathcv3 = "Cvvalentine.pdf"
readercv3 = PdfReader(pdf_pathcv3)
text_pagescv3 = [page.extract_text() for page in readercv3.pages]
full_textcv3 = "\n".join(text_pagescv3)
full_textcv3 = clean_text(full_textcv3)
clean_textcv3 = unidecode.unidecode(full_textcv3)
embeddingscv3 = model.encode(clean_textcv3)
embeddingscv3 = normalize(embeddingscv3.reshape(1, -1))[0]

#### Extraction des embeddings CV agroalim 4

pdf_pathcv4 = "CVcamille.pdf"
readercv4 = PdfReader(pdf_pathcv4)
text_pagescv4 = [page.extract_text() for page in readercv4.pages]
full_textcv4 = "\n".join(text_pagescv4)
full_textcv4 = clean_text(full_textcv4)
clean_textcv4 = unidecode.unidecode(full_textcv4)
embeddingscv4 = model.encode(clean_textcv4)
embeddingscv4 = normalize(embeddingscv4.reshape(1, -1))[0]

#### Extraction des embeddings CV chimie 5

pdf_pathcv5 = "CVchimiste.pdf"
readercv5 = PdfReader(pdf_pathcv5)
text_pagescv5 = [page.extract_text() for page in readercv5.pages]
full_textcv5 = "\n".join(text_pagescv5)
full_textcv5 = full_textcv5.replace("\t", " ")
full_textcv5 = full_textcv5.replace("\n", " ")
#full_textcv7 = full_textcv7.replace("\'", "'")
#full_textcv7 = full_textcv7.replace("*", "")
clean_textcv5 = unidecode.unidecode(full_textcv5)
embeddingscv5 = model.encode(clean_textcv5)
embeddingscv5 = normalize(embeddingscv5.reshape(1, -1))[0]

#### Extraction des embeddings CV commerce 6

pdf_pathcv6 = "Cvcommerce.pdf"
readercv6 = PdfReader(pdf_pathcv6)
text_pagescv6 = [page.extract_text() for page in readercv6.pages]
full_textcv6 = "\n".join(text_pagescv6)
full_textcv6 = clean_text(full_textcv6)
clean_textcv6 = unidecode.unidecode(full_textcv6)
embeddingscv6 = model.encode(clean_textcv6)

#### Extraction des embeddings CV communication 7

pdf_pathcv7 = "Cvcommunication.pdf"
readercv7 = PdfReader(pdf_pathcv7)
text_pagescv7 = [page.extract_text() for page in readercv7.pages]
full_textcv7 = "\n".join(text_pagescv7)
full_textcv7 = clean_text(full_textcv7)
clean_textcv7 = unidecode.unidecode(full_textcv7)
embeddingscv7 = model.encode(clean_textcv7)


#### Extraction des embeddings Offre Agro 1

pdf_pathof1 = "Offretest.pdf"
readerof1 = PdfReader(pdf_pathof1)
text_pagesof1 = [page.extract_text() for page in readerof1.pages]
full_textof1 = "\n".join(text_pagesof1)
full_textof1 = clean_text(full_textof1)
clean_textof1 = unidecode.unidecode(full_textof1)
embeddingsof1 = model.encode(clean_textof1)
embeddingsof1 = normalize(embeddingsof1.reshape(1, -1))[0]


#### Extraction des embeddings Offre Agro Modélisation 2

pdf_pathof2 = "Offremodel.pdf"
readerof2 = PdfReader(pdf_pathof2)
text_pagesof2 = [page.extract_text() for page in readerof2.pages]
full_textof2 = "\n".join(text_pagesof2)
full_textof2 = clean_text(full_textof2)
clean_textof2 = unidecode.unidecode(full_textof2)
embeddingsof2 = model.encode(clean_textof2)
embeddingsof2 = normalize(embeddingsof2.reshape(1, -1))[0]

#### Extraction des embeddings Offre Qualité 3

pdf_pathof3 = "Offrequalité.pdf"
readerof3 = PdfReader(pdf_pathof3)
text_pagesof3 = [page.extract_text() for page in readerof3.pages]
full_textof3 = "\n".join(text_pagesof3)
full_textof3 = clean_text(full_textof3)
clean_textof3 = unidecode.unidecode(full_textof3)
embeddingsof3 = model.encode(clean_textof3)

### Création Dataframe

cv_texts = [clean_textcv1, clean_textcv2, clean_textcv3, clean_textcv4, clean_textcv5]
cv_embeddings = np.array([model.encode(t) for t in cv_texts])
cv_embeddings = normalize(cv_embeddings)

of_texts = [clean_textof1, clean_textof2]
of_embeddings = np.array([model.encode(t) for t in of_texts])
of_embeddings = normalize(of_embeddings)

stackedcv = np.vstack([embeddingscv1, embeddingscv2,embeddingscv3,embeddingscv4,embeddingscv5]) 
stackedof = np.vstack([embeddingsof1, embeddingsof2]) 
datacv = pd.DataFrame(stackedcv)
dataof = pd.DataFrame(stackedof)

###PCA des CV avec projection des offres

pca_cvof = PCA(n_components=2)
x = datacv.values
X_pca = pca_cvof.fit_transform(x)

### projection offre


x2=dataof.values
offers_pca = pca_cvof.transform(x2)

### Graphique

plt.scatter(X_pca[:, 0], X_pca[:, 1], label="CV existants")
for i in range(X_pca.shape[0]):
    plt.text(X_pca[i, 0]+0.01, X_pca[i, 1]+0.01, f'CV {i+1}')
plt.scatter(offers_pca[:, 0], offers_pca[:, 1], label='Offres', color='red')
for j in range(offers_pca.shape[0]):
    plt.text(offers_pca[j, 0]+0.01, offers_pca[j, 1]+0.01, f'Offre {j+1}', color='red')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title('ACP des embeddings de CV')
plt.legend()
plt.show()

###PCA des offres avec projection des cv

X_pca2 = pca_cvof.fit_transform(x2)

### projection offre

offers_pcacv = pca_cvof.transform(x)

### Graphique

plt.scatter(X_pca2[:, 0], X_pca2[:, 1], label="offres existantes")
for i in range(X_pca2.shape[0]):
    plt.text(X_pca2[i, 0]+0.01, X_pca2[i, 1]+0.01, f'Offre {i+1}')
plt.scatter(offers_pcacv[:, 0], offers_pcacv[:, 1], label='CV', color='red')
for j in range(offers_pcacv.shape[0]):
    plt.text(offers_pcacv[j, 0]+0.01, offers_pcacv[j, 1]+0.01, f'CV {j+1}', color='red')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("ACP des embeddings d'offres")
plt.legend()
plt.show()

### PCA Avec embeddings tous du même (même résultats)

of_pca = pca_cvof.fit_transform(of_embeddings)
cv_pca = pca_cvof.transform(cv_embeddings)

plt.figure(figsize=(8,6))

# Offres
plt.scatter(of_pca[:,0], of_pca[:,1], color='red', label='Offres')
for i, txt in enumerate(of_texts):
    plt.text(of_pca[i,0]+0.01, of_pca[i,1]+0.01, f'Offre {i+1}', color='red')

# CV
plt.scatter(cv_pca[:,0], cv_pca[:,1], color='blue', label='CV')
for i, txt in enumerate(cv_texts):
    plt.text(cv_pca[i,0]+0.01, cv_pca[i,1]+0.01, f'CV {i+1}', color='blue')

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Projection PCA des CV et Offres")
plt.legend()
plt.show()

### cosine similarity 

similarities = cosine_similarity(cv_embeddings, of_embeddings)
# similarities[i,j] = similarité entre CV i et offre j

# Exemple : offre la plus proche pour chaque CV
best_matches = similarities.argmax(axis=1)
for i, j in enumerate(best_matches):
    print(f"CV {i+1} correspond le mieux à l'offre {j+1} (similarité={similarities[i,j]:.2f})")

