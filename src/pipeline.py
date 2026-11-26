"""Pipeline complet de sous-titrage automatique."""

from pathlib import Path

from src.transcription import extraire_audio, transcrire, precharger_libs_cuda12
from src.sous_titres import segments_vers_srt, incruster_sous_titres, verifier_srt

def executer_pipeline(chemin_video, taille_modele="small"):
    """Exécute toutes les étapes du sous-titrage automatique."""
    
    precharger_libs_cuda12()   # évite la bascule sur CPU dans Colab

    chemin_video = Path(chemin_video)

    # Vérifier que la vidéo existe
    if not chemin_video.exists():
        raise FileNotFoundError(f"Vidéo introuvable : {chemin_video}")

    # Créer le dossier de sortie s'il n'existe pas
    dossier_sortie = Path("outputs")
    dossier_sortie.mkdir(exist_ok=True)

    # Préparer les chemins des fichiers générés
    chemin_audio = dossier_sortie / f"{chemin_video.stem}.wav"
    chemin_srt = dossier_sortie / f"{chemin_video.stem}.srt"
    chemin_video_finale = dossier_sortie / f"{chemin_video.stem}_sous_titree.mp4"

    print(f"Vidéo à traiter : {chemin_video}")
    print(f"Modèle Whisper : {taille_modele}")
    print("Préparation du traitement terminée.")


    # Étape 1 : extraire le son de la vidéo
    print("\n1. Extraction de l'audio...")
    extraire_audio(chemin_video, chemin_audio)
    print(f"Audio extrait : {chemin_audio}")


    # Étape 2 : transcrire l'audio avec Whisper
    print("\n2. Transcription avec Whisper...")
    segments, duree = transcrire(
        chemin_audio,
        taille_modele=taille_modele
    )

    print(f"Transcription terminée en {duree} secondes.")
    print(f"Nombre de segments détectés : {len(segments)}")


    # Étape 3 : générer le fichier SRT
    print("\n3. Génération des sous-titres...")
    segments_vers_srt(segments, chemin_srt)
    print(f"Fichier SRT créé : {chemin_srt}")

    # Étape 4 : incruster les sous-titres dans la vidéo
    print("\n4. Incrustation des sous-titres dans la vidéo...")
    incruster_sous_titres(
        chemin_video,
        chemin_srt,
        chemin_video_finale
    )

    print(f"Vidéo finale créée : {chemin_video_finale}")
    verifier_srt(chemin_srt)

    return {
        "video": str(chemin_video_finale),
        "srt": str(chemin_srt),
        "duree_transcription": duree,
        "nb_segments": len(segments),
    }
