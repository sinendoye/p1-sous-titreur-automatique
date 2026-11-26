"""Module 1 — Extraction du son et transcription (faster-whisper + VAD).

Format des segments renvoyés (convenu avec l'équipe) :
    [{"start": 0.0, "end": 3.2, "text": "..."}, ...]
"""
import ctypes
import glob
import os
import subprocess
import time

import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel


def precharger_libs_cuda12():
    """Contourne l'erreur 'libcublas.so.12 not found' sur Colab."""
    try:
        import nvidia.cublas, nvidia.cudnn
    except ImportError:
        print("Installer d'abord : pip install nvidia-cublas-cu12 'nvidia-cudnn-cu12==9.*'")
        return
    restants = []
    for paquet in (nvidia.cublas, nvidia.cudnn):
        restants += sorted(glob.glob(os.path.join(paquet.__path__[0], "lib", "lib*.so*")))
    for _ in range(3):
        echecs = []
        for f in restants:
            try:
                ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                echecs.append(f)
        restants = echecs
        if not restants:
            break


def extraire_audio(chemin_video, chemin_audio=None):
    """Extrait le son d'une vidéo en WAV mono 16 kHz (Colab, Linux ou Windows)."""
    if chemin_audio is None:
        import tempfile
        chemin_audio = os.path.join(tempfile.gettempdir(), "audio.wav")

    commande = [
        "ffmpeg", "-y",
        "-i", str(chemin_video),
        "-vn", "-ac", "1", "-ar", "16000",
        str(chemin_audio),
    ]
    subprocess.run(commande, check=True, capture_output=True)
    return chemin_audio


def charger_audio(chemin_audio):
    """Lit un WAV 16 kHz et le convertit en nombres décimaux entre -1 et 1."""
    frequence, donnees = wavfile.read(chemin_audio)
    assert frequence == 16000, f"Fréquence inattendue : {frequence}"
    return donnees.astype(np.float32) / 32768.0


_MODELES = {}  # cache : chaque modèle n'est chargé qu'une seule fois par appareil


def charger_modele(taille_modele, device="cuda"):
    """Charge un modèle Whisper sur le GPU (cuda) ou le processeur (cpu)."""
    cle = (taille_modele, device)
    if cle not in _MODELES:
        compute_type = "float16" if device == "cuda" else "int8"
        _MODELES[cle] = WhisperModel(taille_modele, device=device, compute_type=compute_type)
    return _MODELES[cle]


def transcrire(chemin_audio, taille_modele="small", vocabulaire=None):
    """Transcrit un audio en français. Renvoie (segments, durée de calcul en secondes).

    Essaie d'abord le GPU, puis bascule sur le CPU si le GPU est indisponible.
    """
    audio = charger_audio(chemin_audio)
    for device in ("cuda", "cpu"):
        try:
            modele = charger_modele(taille_modele, device)
            debut = time.time()
            segments_bruts, _ = modele.transcribe(
                audio, language="fr", vad_filter=True, beam_size=5, initial_prompt=vocabulaire
            )
            segments = [
                {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
                for s in segments_bruts
            ]
            return segments, round(time.time() - debut, 2)
        except RuntimeError as erreur:
            if device == "cpu":
                raise  # même le CPU a échoué : on signale l'erreur
            print(f"⚠️ GPU indisponible ({erreur}). Passage sur CPU, plus lent.")


def transcrire_video(chemin_video, taille_modele="small", vocabulaire=None):
    """Raccourci : vidéo → segments, en une seule ligne."""
    return transcrire(extraire_audio(chemin_video), taille_modele, vocabulaire)
