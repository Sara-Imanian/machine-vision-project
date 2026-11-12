import cv2
from pathlib import Path


SRC_DIR = Path("data_original/leaves")
DST_DIR = Path("data/leaves")
MAX_SIDE = 1024

def prepare_images():
    count = 0
    for src_path in sorted(SRC_DIR.rglob("*.jpg")):
        img = cv2.imread(str(src_path))
        if img is None:
            print(f"skip {src_path.name}")
            continue

        h, w = img.shape[:2]
        scale = MAX_SIDE / max(h, w)
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        out_path = DST_DIR / src_path.relative_to(SRC_DIR)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        count +=1

    print(f'done: {count} images')


if __name__ == '__main__':
    prepare_images()

