drive.mount('/content/drive')

zip_dir = '/content/drive/MyDrive/data/Comprehensive Bangla TTS.zip'
extract_dir = '/content/drive/MyDrive/data/BanglaTTS/'

os.makedirs(extract_dir, exist_ok=True)

# Extracting .zip dataset
print("Unzipping... this may take a few minutes.")
with zipfile.ZipFile(zip_dir, 'r') as z:
    z.extractall(extract_dir)
print("Done!")

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


