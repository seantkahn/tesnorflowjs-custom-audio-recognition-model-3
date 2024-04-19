#pip install pydub
#run instructions: python script_name.py path_to_your_file.wav

import sys
from pydub import AudioSegment
from pydub.silence import split_on_silence

def split_letters(audio_file, min_silence_len=500, silence_thresh=-40, keep_silence=100):
    """
    Splits a WAV file into multiple segments based on silence.

    Args:
    audio_file (str): Path to the WAV file.
    min_silence_len (int): Minimum length of silence in milliseconds that is used to split the audio.
    silence_thresh (int): Silence threshold in dB. Lower values mean more silence will be considered.
    keep_silence (int): Amount of silence in milliseconds to keep at the beginning and end of each split.

    Returns:
    List of AudioSegments.
    """
    # Load the audio file
    sound = AudioSegment.from_wav(audio_file)

    # Split on silence
    audio_chunks = split_on_silence(
        sound,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )

    # Save each letter as a separate file
    for i, chunk in enumerate(audio_chunks):
        out_file = f"letter_{chr(65 + i)}.wav"  # Naming files as letter_A.wav, letter_B.wav, etc.
        print(f"Exporting {out_file}...")
        chunk.export(out_file, format="wav")
    
    return audio_chunks

def main():
    if len(sys.argv) != 2:
        print("Usage: python script_name.py path_to_your_file.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    split_letters(audio_file)

if __name__ == "__main__":
    main()
