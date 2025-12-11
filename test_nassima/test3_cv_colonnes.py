
import pdfplumber
import re

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
        "Experience": ["experience", "expériences", "parcours", "postes", "emploi", "expériences professionelles"],
        "Formation": ["formation", "education", "diplomes", "études", "cursus"],
        "Competences": ["competences", "hard skills", "outils", "langues", "logiciels", "atouts", "compétences techniques"]
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

# --- TEST ---

fichier_cv = "CV__Nassima_El_Hilali.pdf"
resultat = extraire_cv_colonnes(fichier_cv, ratio_colonne=0.30) # Essayez 0.30 ou 0.35

if resultat:
    print(f"--- ANALYSE DE {fichier_cv} ---")
    print("\n🟦 COMPÉTENCES DÉTECTÉES :")
    print(resultat['Competences'][:500] + "...") # Affiche le début
    
    print("\n🟨 EXPÉRIENCE DÉTECTÉE :")
    print(resultat['Experience'][:500] + "...")
    
    print("\n🟩 FORMATION DÉTECTÉE :")
    print(resultat['Formation'][:500] + "...")