drive.mount('/content/drive')

zip_dir = '/content/drive/MyDrive/data/Comprehensive Bangla TTS.zip'
extract_dir = '/content/drive/MyDrive/data/BanglaTTS/'

os.makedirs(extract_dir, exist_ok=True)

'''
# Extracting .zip dataset
print("Unzipping... this may take a few minutes.")
with zipfile.ZipFile(zip_dir, 'r') as z:
    z.extractall(extract_dir)
print("Done!")
'''

# Viewing extracted files
for root, dirs, files in os.walk(extract_dir):
    depth = root.replace(extract_dir, '').count(os.sep)
    if depth > 2:
        continue
    indent = '  ' * depth
    print(f"{indent}{os.path.basename(root)}/")
    for f in files[:5]:
        print(f"  {indent}{f}")
    if len(files) > 5:
        print(f"  {indent}... and {len(files)-5} more")

DATA_DIR = '/content/drive/MyDrive/data/BanglaTTS/iitm_bangla_tts/comprehensive_bangla_tts'

# Explore audio data folders
for speaker in ['female', 'male']:
    speaker_dir = os.path.join(DATA_DIR, speaker)
    print(f"\n=== {speaker.upper()} FOLDER ===")

    all_files = list(Path(speaker_dir).rglob("*.*"))

    # Show file type breakdown
    from collections import Counter
    ext_counts = Counter(f.suffix.lower() for f in all_files)
    print("File types:", dict(ext_counts))

    # Show first 10 files
    for f in sorted(all_files)[:10]:
        print(f"  {f.relative_to(DATA_DIR)}")

    if len(all_files) > 10:
        print(f"  ... and {len(all_files)-10} more")

# Read the transcript files
for speaker in ['female', 'male']:
    print(f"\n=== {speaker.upper()} — metadata.txt (first 10 lines) ===")
    meta_path = os.path.join(DATA_DIR, speaker, 'mono', f'metadata_{speaker}.txt')
    with open(meta_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            print(repr(line.strip()))  # repr() so we can see the exact delimiter
            if i >= 9:
                break

    print(f"\n=== {speaker.upper()} — txt.done.data (first 10 lines) ===")
    data_path = os.path.join(DATA_DIR, speaker, 'mono', 'txt.done.data')
    with open(data_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            print(repr(line.strip()))
            if i >= 9:
                break

# Audio duration and sample rate check
for speaker in ['female', 'male']:
    wav_dir = Path(DATA_DIR) / speaker / 'mono' / 'wav'
    all_wavs = list(wav_dir.glob('*.wav'))

    # Sample 100 random files for speed
    sample = random.sample(all_wavs, min(100, len(all_wavs)))

    durations = []
    sr_set = set()
    for wav in sample:
        y, sr = librosa.load(str(wav), sr=None)
        durations.append(len(y) / sr)
        sr_set.add(sr)

    durations = np.array(durations)
    print(f"\n=== {speaker.upper()} ===")
    print(f"Sample rate(s) found : {sr_set}")
    print(f"Duration — min       : {durations.min():.2f}s")
    print(f"Duration — max       : {durations.max():.2f}s")
    print(f"Duration — mean      : {durations.mean():.2f}s")
    print(f"Duration — median    : {np.median(durations):.2f}s")
    print(f"Clips under 1s       : {(durations < 1).sum()}")
    print(f"Clips over 10s       : {(durations > 10).sum()}")
