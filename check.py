import os, glob

folder = "data/processed"
files = sorted(glob.glob(os.path.join(folder, "train_*.csv")))

# Sort naturally by number after "train_"
def get_index(f):
    base = os.path.basename(f)
    num = base.replace("train_", "").replace(".csv", "")
    return int(num) if num.isdigit() else 999999

files = sorted(files, key=get_index)

print(f"Found {len(files)} files before renaming.")
new_index = 1

for old_path in files:
    new_name = f"train_{new_index}.csv"
    new_path = os.path.join(folder, new_name)
    os.rename(old_path, new_path)
    print(f"Renamed: {os.path.basename(old_path)} → {new_name}")
    new_index += 1

print(f"\n✅ Renaming complete. Now you have {new_index-1} sequential files.")