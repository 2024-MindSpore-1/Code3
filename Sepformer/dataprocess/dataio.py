import csv
import re
import numpy as np
from scipy.io.wavfile import read


def load_data_csv(csv_path, replacements={}):
    """Loads CSV and formats string values.

    Uses the SpeechBrain legacy CSV data format, where the CSV must have an
    'ID' field.
    If there is a field called duration, it is interpreted as a float.
    The rest of the fields are left as they are (legacy _format and _opts fields
    are not used to load the data in any special way).

    Bash-like string replacements with $to_replace are supported.

    Arguments
    ----------
    csv_path : str
        Path to CSV file.
    replacements : dict
        (Optional dict), e.g., {"data_folder": "/home/speechbrain/data"}
        This is used to recursively format all string values in the data.

    Returns
    -------
    dict
        CSV data with replacements applied.
    """

    with open(csv_path, newline="") as csvfile:
        result = {}
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        variable_finder = re.compile(r"\$([\w.]+)")
        for row in reader:
            # ID:
            try:
                data_id = row["ID"]
                del row["ID"]  # This is used as a key in result, instead.
            except KeyError:
                raise KeyError(
                    "CSV has to have an 'ID' field, with unique ids"
                    " for all data points"
                )
            if data_id in result:
                raise ValueError(f"Duplicate id: {data_id}")
            # Replacements:
            for key, value in row.items():
                try:
                    row[key] = variable_finder.sub(
                        lambda match: str(replacements[match[1]]), value
                    )
                except KeyError:
                    raise KeyError(
                        f"The item {value} requires replacements "
                        "which were not supplied."
                    )
            # Duration:
            if "duration" in row:
                row["duration"] = float(row["duration"])
            result[data_id] = row
    return result


def read_audio(waveforms_obj):
    """
    return a numpy array.
    """
    if isinstance(waveforms_obj, str):
        audio, _ = _load_audio(waveforms_obj)
        audio = audio.astype(np.float32)
        return audio
    path = waveforms_obj["file"]
    start = waveforms_obj.get("start", 0)
    # Default stop to start -> if not specified, num_frames becomes 0,
    # which is the torchaudio default
    stop = waveforms_obj.get("stop", start)
    audio, fs = _load_audio(path)
    audio = audio.astype(np.float32)
    return audio[start:stop]


def _load_audio(wave_path):
    sr, wav = read(wave_path)
    if len(wav.shape) == 2:
        wav = wav[:, 0]
    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0
    return wav, sr
