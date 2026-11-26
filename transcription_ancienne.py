"""Module 1 ÔÇö Extraction du son et transcription (faster-whisper + VAD).

Format des segments renvoy├®s (convenu avec l'├®quipe) :
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


def extraire_audio(chemin_video, chemin_audio="/content/audio.wav"):
    """Extrait le son d'une vid├®o en WAV mono 16 kHz."""
    commande = ["ffmpeg", "-y", "-i", chemin_video, "-vn", "-ac", "1", "-ar", "16000", chemin_audio]
    subprocess.run(commande, check=True, capture_output=True)
    return chemin_audio


def charger_audio(chemin_audio):
    """Lit un WAV 16 kHz et le convertit en nombres d├®cimaux entre -1 et 1."""
    frequence, donnees = wavfile.read(chemin_audio)
    assert frequence == 16000, f"Fr├®quence inattendue : {frequence}"
    return donnees.astype(np.float32) / 32768.0


_MODELES = {}  # cache : chaque mod├¿le n'est charg├® qu'une seule fois


def charger_modele(taille_modele):
    if taille_modele not in _MODELES:
        _MODELES[taille_modele] = WhisperModel(taille_modele, device="cuda", compute_type="float16")
    return _MODELES[taille_modele]


def transcrire(chemin_audio, taille_modele="small", vocabulaire=None):
    """Transcrit un audio en fran├ºais. Renvoie (segments, dur├®e de calcul en secondes)."""
    modele = charger_modele(taille_modele)
    audio = charger_audio(chemin_audio)
    debut = time.time()
    segments_bruts, _ = modele.transcribe(
        audio, language="fr", vad_filter=True, beam_size=5, initial_prompt=vocabulaire
    )
    segments = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segments_bruts
    ]
    return segments, round(time.time() - debut, 2)


def transcrire_video(chemin_video, taille_modele="small", vocabulaire=None):
    """Raccourci : vid├®o ÔåÆ segments, en une seule ligne."""
    return transcrire(extraire_audio(chemin_video), taille_modele, vocabulaire)
