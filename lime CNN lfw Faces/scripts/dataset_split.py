import os, shutil, random
from pathlib import Path

SOURCE_DIR = Path("data/raw/lfw_funneled")
TARGET_DIR = Path("data/processed")
SPLIT = {"train": 0.7, "val": 0.15, "test": 0.15}

for person_dir in SOURCE_DIR.iterdir():
    images = list(person_dir.glob("*.jpg"))
    random.shuffle(images)
    n = len(images)
    parts = {
        "train": images[:int(n * SPLIT["train"])],
        "val": images[int(n * SPLIT["train"]):int(n * (SPLIT["train"] + SPLIT["val"]))],
        "test": images[int(n * (SPLIT["train"] + SPLIT["val"])):]
    }
    for split, files in parts.items():
        out_dir = TARGET_DIR / split / person_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, out_dir)
