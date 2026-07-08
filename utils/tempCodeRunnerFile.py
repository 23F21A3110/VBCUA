import librosa
import numpy as np

def fluency_score(audio_path):

    y, sr = librosa.load(audio_path)

    duration = librosa.get_duration(
        y=y,
        sr=sr
    )

    rms = np.mean(
        librosa.feature.rms(y=y)
    )

    silence = np.sum(
        np.abs(y) < 0.01
    )

    pause_ratio = silence / len(y)

    return {
        "duration": round(duration, 2),
        "energy": round(float(rms), 3),
        "pause_ratio": round(float(pause_ratio), 2)
    }