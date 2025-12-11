
def decouper_cv_en_blocs(texte):
    texte = texte.lower()

    blocs = {
        "competences": "",
        "experiences": "",
        "formations": ""
    }

    # Mots-clés possibles
    keys = {
        "competences": ["competences", "skills"],
        "experiences": ["experience", "experiences", "projets"],
        "formations": ["formation", "education", "etudes"]
    }

    lignes = texte.split(" ")

    current_bloc = None

    for mot in lignes:
        for bloc, mots in keys.items():
            if mot in mots:
                current_bloc = bloc
        if current_bloc:
            blocs[current_bloc] += mot + " "

    return blocs