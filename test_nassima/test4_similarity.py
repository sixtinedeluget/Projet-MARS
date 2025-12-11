# -*- coding: utf-8 -*-
"""
Comparaison similarité :
- CV brut vs Offre
- CV découpé (blocs) vs Offre

@author: Nassima
"""

import os
import unidecode
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 1. Nettoyage texte
# =========================

def clean_text(txt):
    replacements = {
        "*": "",
        "’": "'",
        "‘": "'",
        "\t": " ",
        "\n": " ",
        "\r": " ",
    }
    for old, new in replacements.items():
        txt = txt.replace(old, new)
    return txt.strip()


# =========================
# 2. Découpage en blocs par mots-clés
# =========================

def segmenter_texte_par_mots_cles(texte):
    sections = {
        "Experience": [],
        "Formation": [],
        "Competences": [],
        "Autre": []
    }

    keywords = {
        "Experience": ["experience", "expériences", "parcours", "postes", "emploi"],
        "Formation": ["formation", "education", "diplomes", "études", "cursus"],
        "Competences": ["competences", "hard skills", "outils", "langues", "logiciels"]
    }

    current_section = "Autre"

    lignes = texte.split('\n')
    for ligne in lignes:
        ligne_clean = ligne.strip()
        if not ligne_clean:
            continue

        ligne_lower = ligne_clean.lower()

        if len(ligne_clean) < 40:
            for section, triggers in keywords.items():
                if any(trigger in ligne_lower for trigger in triggers):
                    current_section = section
                    break

        sections[current_section].append(ligne_clean)

    return sections


# =========================
# 3. Découpage CV EN 2 COLONNES
# =========================

def extraire_cv_colonnes(pdf_path, ratio_colonne=0.30):
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

                bbox_gauche = (0, 0, width * ratio_colonne, height)
                bbox_droite = (width * ratio_colonne, 0, width, height)

                zone_gauche = page.crop(bbox_gauche)
                zone_droite = page.crop(bbox_droite)

                texte_gauche = zone_gauche.extract_text() or ""
                texte_droite = zone_droite.extract_text() or ""

                dict_gauche = segmenter_texte_par_mots_cles(texte_gauche)
                dict_droite = segmenter_texte_par_mots_cles(texte_droite)

                for key in contenu_final:
                    contenu_final[key].extend(dict_gauche[key])
                    contenu_final[key].extend(dict_droite[key])

        return {k: " ".join(v) for k, v in contenu_final.items()}

    except Exception as e:
        print(f"❌ Erreur avec pdfplumber sur {pdf_path}: {e}")
        return None


# =========================
# 4. EMBEDDING CV BRUT
# =========================

def embedding_cv_brut(pdf_path, model):
    texte = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            texte += page.extract_text() + "\n"

    texte = clean_text(texte)
    texte = unidecode.unidecode(texte)

    emb = model.encode(texte)
    emb = normalize(emb.reshape(1, -1))[0]
    return emb


# =========================
# 5. EMBEDDING CV DÉCOUPÉ (MOYENNE DES BLOCS)
# =========================

def embedding_cv_decoupe(pdf_path, model):
    resultat = extraire_cv_colonnes(pdf_path, ratio_colonne=0.30)

    # Sécurité si jamais un bloc est vide
    texte_comp = resultat["Competences"] if resultat["Competences"] else " "
    texte_exp  = resultat["Experience"] if resultat["Experience"] else " "
    texte_form = resultat["Formation"] if resultat["Formation"] else " "

    # Nettoyage
    texte_comp = unidecode.unidecode(clean_text(texte_comp))
    texte_exp  = unidecode.unidecode(clean_text(texte_exp))
    texte_form = unidecode.unidecode(clean_text(texte_form))

    # 1️ Embedding de chaque bloc séparément
    emb_comp = model.encode(texte_comp)
    emb_exp  = model.encode(texte_exp)
    emb_form = model.encode(texte_form)

    # 2️ MOYENNE MATHÉMATIQUE DES 3 VECTEURS
    emb_final = (emb_comp + emb_exp + emb_form) / 3

    # 3️ Normalisation
    emb_final = normalize(emb_final.reshape(1, -1))[0]

    return emb_final

# =========================
# 6. EMBEDDING OFFRE
# =========================

def embedding_offre(pdf_path, model):
    texte = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            texte += page.extract_text() + "\n"

    texte = clean_text(texte)
    texte = unidecode.unidecode(texte)

    emb = model.encode(texte)
    emb = normalize(emb.reshape(1, -1))[0]
    return emb


# =========================
# 7. CHEMINS AUTOMATIQUES
# =========================

BASE_DIR = os.path.dirname(__file__)

cv_path = r"C:\Users\dell\Documents\cours M2\projet_ing\Projet-MARS\test_nassima\CV__Nassima_EL_HILALI.pdf"
offre_path = r"C:\Users\dell\Documents\cours M2\projet_ing\Projet-MARS\Offre\Offretest.pdf"


# =========================
# 8. CALCUL DES SIMILARITÉS
# =========================

model = SentenceTransformer("all-mpnet-base-v2")

emb_cv_brut = embedding_cv_brut(cv_path, model)
emb_cv_decoupe = embedding_cv_decoupe(cv_path, model)
emb_offre = embedding_offre(offre_path, model)

sim_brut = cosine_similarity([emb_cv_brut], [emb_offre])[0][0]
sim_decoupe = cosine_similarity([emb_cv_decoupe], [emb_offre])[0][0]


# =========================
# 9. AFFICHAGE FINAL
# =========================

print("\n===============================")
print(" CV  → Offre")
print("===============================")
print(f" Similarité CV BRUT     : {sim_brut:.3f}")
print(f" Similarité CV DÉCOUPÉ  : {sim_decoupe:.3f}")
print("===============================\n")