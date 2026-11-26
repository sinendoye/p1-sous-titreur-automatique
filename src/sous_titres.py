import textwrap
import subprocess
import shutil
from pathlib import Path


def _secondes_vers_srt(secondes):
    """Convertit des secondes en format HH:MM:SS,mmm."""

    millisecondes = round(secondes * 1000)

    heures, reste = divmod(millisecondes, 3600000)
    minutes, reste = divmod(reste, 60000)
    secondes_entieres, millisecondes = divmod(reste, 1000)

    return (
        f"{heures:02d}:"
        f"{minutes:02d}:"
        f"{secondes_entieres:02d},"
        f"{millisecondes:03d}"
    )


def segments_vers_srt(segments, chemin_srt):
    """Transforme des segments Whisper en fichier SRT."""

    blocs = []
    numero = 1

    for segment in segments:

        debut = float(segment["start"])
        fin = float(segment["end"])
        texte = segment["text"].strip()

        if not texte:
            continue

        if fin < debut:
            raise ValueError("La fin du segment précède son début.")

        # Découper le texte en lignes de 42 caractères maximum
        lignes = textwrap.wrap(texte, width=42)

        # Regrouper les lignes deux par deux
        groupes = [
            lignes[i:i + 2]
            for i in range(0, len(lignes), 2)
        ]

        # Répartir la durée selon le nombre de caractères
        longueurs = [len(" ".join(groupe)) for groupe in groupes]
        total_caracteres = sum(longueurs)

        duree = fin - debut
        caracteres_cumules = 0

        for i, groupe in enumerate(groupes):

            debut_sous_titre = (
                debut + duree * caracteres_cumules / total_caracteres
            )

            caracteres_cumules += longueurs[i]

            if i == len(groupes) - 1:
                fin_sous_titre = fin
            else:
                fin_sous_titre = (
                    debut + duree * caracteres_cumules / total_caracteres
                )

            texte_sous_titre = "\n".join(groupe)

            bloc = (
                f"{numero}\n"
                f"{_secondes_vers_srt(debut_sous_titre)} --> "
                f"{_secondes_vers_srt(fin_sous_titre)}\n"
                f"{texte_sous_titre}"
            )

            blocs.append(bloc)
            numero += 1

    # Écrire tous les sous-titres dans le fichier SRT
    with open(chemin_srt, "w", encoding="utf-8") as fichier:
        fichier.write("\n\n".join(blocs) + "\n")



def incruster_sous_titres(chemin_video, chemin_srt, chemin_sortie):
    """Incruste les sous-titres SRT dans une vidéo avec FFmpeg."""

    # Vérifier que FFmpeg est disponible
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("FFmpeg n'est pas installé ou accessible.")

    # Transformer les chemins en objets Path
    video = Path(chemin_video)
    srt = Path(chemin_srt)
    sortie = Path(chemin_sortie)

    # Vérifier que les fichiers d'entrée existent
    if not video.is_file():
        raise FileNotFoundError(f"Vidéo introuvable : {video}")

    if not srt.is_file():
        raise FileNotFoundError(f"Fichier SRT introuvable : {srt}")

    # Préparer le chemin du fichier SRT pour FFmpeg
    chemin_filtre = srt.resolve().as_posix()
    chemin_filtre = chemin_filtre.replace(":", r"\:")
    chemin_filtre = chemin_filtre.replace("'", r"\'")

    # Construire la commande FFmpeg
    commande = [
        "ffmpeg",
        "-y",
        "-i", str(video),
        "-vf", f"subtitles='{chemin_filtre}'",
        "-c:a", "copy",
        str(sortie)
    ]

    # Exécuter FFmpeg
    subprocess.run(commande, check=True)

    print(f"Vidéo sous-titrée créée : {sortie}")


def verifier_srt(chemin_srt):
    """Vérifie les règles de longueur et de nombre de lignes d'un SRT."""

    chemin_srt = Path(chemin_srt)

    if not chemin_srt.is_file():
        raise FileNotFoundError(f"Fichier SRT introuvable : {chemin_srt}")

    with open(chemin_srt, "r", encoding="utf-8") as fichier:
        contenu = fichier.read().strip()

    erreurs = []

    blocs = contenu.split("\n\n")

    for numero_bloc, bloc in enumerate(blocs, start=1):
        lignes = bloc.splitlines()

        if len(lignes) < 3:
            erreurs.append(
                f"Bloc {numero_bloc} : bloc SRT incomplet."
            )
            continue

        lignes_texte = lignes[2:]

        if len(lignes_texte) > 2:
            erreurs.append(
                f"Bloc {numero_bloc} : plus de 2 lignes de texte."
            )

        for numero_ligne, ligne in enumerate(lignes_texte, start=1):
            if len(ligne) > 42:
                erreurs.append(
                    f"Bloc {numero_bloc}, ligne {numero_ligne} : "
                    f"{len(ligne)} caractères (maximum 42)."
                )

    if erreurs:
        print("Des problèmes ont été détectés :")
        for erreur in erreurs:
            print("-", erreur)
        return False

    print("Fichier SRT valide : toutes les règles sont respectées.")
    return True