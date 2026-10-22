import random, shutil
from pathlib import Path

random.seed(42)
src = Path("data_raw/PlantVillage")
out = Path("data")

classes = [c for c in sorted(src.iterdir()) if c.is_dir()]
print(f"Found {len(classes)} classes")

for cls in classes:
    files = [f for f in cls.iterdir() if f.is_file()]
    random.shuffle(files)
    n = len(files)
    n_train, n_val = int(n * 0.70), int(n * 0.15)
    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }
    for name, items in splits.items():
        dest = out / name / cls.name
        dest.mkdir(parents=True, exist_ok=True)
        for f in items:
            shutil.move(str(f), dest / f.name)

print("Done")