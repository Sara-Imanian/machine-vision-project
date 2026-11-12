import numpy as np
from pathlib import Path

from src.preprocess import load_and_preprocess, segment_leaf, extract_all_features




def rename_images(data_dir="data/leaves", dry_run=False):
    for class_dir in sorted(Path(data_dir).iterdir()):
        if not class_dir.is_dir():
            continue

        for i, img_path in enumerate((sorted(class_dir.glob("*.jpg"))), start=1):
            new_name = f'{class_dir.name}_{i:03d}_a{img_path.suffix}'
            print(img_path.name, "->", new_name)

            if not dry_run:
                img_path.rename(class_dir / new_name)




def load_dataset(data_dir="data/leaves"):
    X, y, groups = [], [], []
    skipped = 0

    for class_dir in sorted(Path(data_dir).iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.glob("*.jpg")):
            try:
                img, gray, blur = load_and_preprocess(img_path)
                mask, contour = segment_leaf(blur)
                if  contour is None:
                    skipped += 1
                    continue
            

                features = extract_all_features(img, gray, mask, contour)
                X.append(features)
                y.append(class_dir.name)
                groups.append(img_path.stem.rsplit("_", 1)[0])

            except Exception as e:
                print(f'skip {img_path.name}: {e}')
                skipped += 1

    print(f"loaded: {len(X)}  |  skipped: {skipped}")
    return  np.array(X), np.array(y), np.array(groups)







if __name__ == "__main__" :

    X, y, groups = load_dataset("data/leaves")
    # rename_images("data/leaves", dry_run=False)
    print(X.shape, y.shape, groups.shape)
    print(set(y.tolist()))
    print(set(groups))
    

    

