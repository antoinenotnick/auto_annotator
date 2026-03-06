from pathlib import Path
import cv2
import numpy as np

from sam_segmentation import SAMSegmenter, COCOExporter
from sam_segmentation.utils import load_image_with_exif

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent

# Object to segment
OBJECT = "insulator"

# Visual prompt: bounding box in pixel coordinates (x, y, w, h), starting from top left corner
# box_prompt = [1000.0, 1550.0, 3000.0, 850.0] # pole in image 1pole-down.jpg
# box_prompt = [500.0, 250.0, 25.0, 60.0] # insulator in image pole_ex2.jpg
box_prompt = [1500.0, 1200.0, 500.0, 350.0] # area in image image2.JPG


def single_image_processing():
    """Process a single image."""
    print("\n" + "=" * 60)
    print("Single Image Processing")
    print("=" * 60)

    segmenter = SAMSegmenter(
        text_prompt = None,
        box_prompt=box_prompt,
        category_name=OBJECT
    )
    result = segmenter.process_image(
        SCRIPT_DIR / "images" / "image2.JPG",
        output_dir=SCRIPT_DIR / "output"
        ) # sample, change if needed

    print(f"Image: {result.image_path.name}")
    print(f"Size: {result.image_size[0]}x{result.image_size[1]}")
    print(f"Detections: {result.num_detections}")
    print(f"Scores: {result.scores}")

    return result


def simple_batch_processing():
    """Simple batch processing with default settings."""
    print("=" * 60)
    print("Simple Batch Processing")
    print("=" * 60)

    segmenter = SAMSegmenter(
        text_prompt = None,
        box_prompt=box_prompt,
        category_name=OBJECT
    )
    result = segmenter.process_directory(
        SCRIPT_DIR / "images",
        output_dir=SCRIPT_DIR / "output",
    )

    print(f"\nProcessed {len(result)} images")
    for result in result:
        print(f"  - {result.image_path.name}: {result.num_detections} detections")

    return result 


def box_prompt_batch_processing():
    """
    Use a single visual box prompt to search a batch of images.

    Args:
        images_dir: Directory of images to scan (default: SCRIPT_DIR / \"images\").
        box: Box prompt (x, y, w, h) in pixel coordinates. If None, uses global box_prompt.
        category_name: Category label to assign to detections from the box prompt.

    Returns:
        List of SegmentationResult objects for images that had at least one detection.
    """
    segmenter = SAMSegmenter(
        text_prompt=None,
        box_prompt=box_prompt,
        category_name=OBJECT,
        save_overlay=True,
    )

    result = segmenter.process_directory(
        SCRIPT_DIR / "images",
        output_dir=SCRIPT_DIR / "output",
    )

    return result


def _compute_mask_histogram(image_array: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """
    Compute a color histogram for the region defined by a mask.

    Returns a 1D normalized histogram vector, or None if the mask is empty.
    """
    if mask.ndim > 2:
        mask = mask.squeeze()

    mask_bool = mask > 0.5
    if not mask_bool.any():
        return None

    y_indices, x_indices = np.where(mask_bool)
    y_min, y_max = int(y_indices.min()), int(y_indices.max())
    x_min, x_max = int(x_indices.min()), int(x_indices.max())

    patch = image_array[y_min : y_max + 1, x_min : x_max + 1]
    patch_mask = mask_bool[y_min : y_max + 1, x_min : x_max + 1].astype("uint8") * 255

    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], patch_mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def find_similar_objects_by_example(
    ref_image: Path | None = None,
    hist_threshold: float = 0.7,
):
    """
    2‑stage workflow:
      1) Use a box prompt on a reference image to get a mask for the example object.
      2) Use a text prompt on all images and keep only detections whose appearance
         (color histogram) is similar to the example mask.

    Args:
        ref_image: Reference image containing the object you drew the box_prompt on.
                   Defaults to the sample image in images/.
        hist_threshold: Similarity threshold (CORREL) in [‑1, 1]; higher = stricter.

    Returns:
        List of (SegmentationResult, best_similarity) for images that contain at
        least one similar object.
    """
    if ref_image is None:
        ref_image = SCRIPT_DIR / "images" / "1pole-down.jpg"

    # 1) Get reference mask from visual (box) prompt on the reference image
    ref_segmenter = SAMSegmenter(
        text_prompt=None,
        box_prompt=box_prompt,
        category_name=OBJECT,
        save_overlay=True,
    )
    ref_result = ref_segmenter.process_image(ref_image, output_dir=SCRIPT_DIR / "output")

    if ref_result.num_detections == 0:
        print("No detections found in reference image with the given box_prompt.")
        return []

    # Use the highest‑scoring mask from the reference image as the exemplar
    ref_scores = ref_result.scores if ref_result.scores is not None else np.zeros(len(ref_result.masks))
    best_idx = int(ref_scores.argmax())
    ref_mask = ref_result.masks[best_idx]

    ref_img = load_image_with_exif(ref_result.image_path, enable_exif=True)
    ref_img_arr = np.array(ref_img.convert("RGB"))
    ref_hist = _compute_mask_histogram(ref_img_arr, ref_mask)
    if ref_hist is None:
        print("Reference mask is empty; cannot compute exemplar appearance.")
        return []

    # 2) Run text‑prompt segmentation on all images, then filter by similarity to exemplar
    batch_segmenter = SAMSegmenter(
        text_prompt=OBJECT,
        box_prompt=None,
        category_name=OBJECT,
        save_overlay=True,
    )
    all_results = batch_segmenter.process_directory(
        SCRIPT_DIR / "images",
        output_dir=SCRIPT_DIR / "output",
    )

    matches: list[tuple[object, float]] = []

    for res in all_results:
        # Skip the reference image itself (optional)
        if res.image_path == ref_result.image_path:
            continue

        img = load_image_with_exif(res.image_path, enable_exif=True)
        img_arr = np.array(img.convert("RGB"))

        best_sim = -1.0
        for mask in res.masks:
            hist = _compute_mask_histogram(img_arr, mask)
            if hist is None:
                continue
            sim = cv2.compareHist(ref_hist.astype("float32"), hist.astype("float32"), cv2.HISTCMP_CORREL)
            if sim > best_sim:
                best_sim = sim

        if best_sim >= hist_threshold:
            matches.append((res, best_sim))

    print(f"\nFound {len(matches)} images with at least one object similar to the exemplar.")
    for res, sim in matches:
        print(f"  - {res.image_path.name}: best similarity {sim:.3f}")

    return matches


def export_to_coco(results):
    exporter = COCOExporter(
        category_name=OBJECT,
        dataset_name="My Dataset"
    )
    exporter.export(results, "annotations.json")


def custom_export(): # Segments and exports to COCO json
    """Process without exports, then export with custom settings."""
    print("\n" + "=" * 60)
    print("Custom Export")
    print("=" * 60)

    # Process without automatic exports (pass empty list)
    segmenter = SAMSegmenter(
        text_prompt=OBJECT,
        export_format=[],  # No automatic exports
        save_overlay=True,
    )

    results = segmenter.process_directory(SCRIPT_DIR / "images")

    # Custom COCO export with different settings
    exporter = COCOExporter(
            category_name=OBJECT,
        dataset_name=OBJECT + " dataset",
        polygon_tolerance=1.5,  # More precise polygons
    )
    exporter.export(results, SCRIPT_DIR / "output" / "custom_coco.json")
    print("Custom COCO export complete")


def main():
# Uncomment any function that you would like to use

    single_image_processing()
    # simple_batch_processing()
    # box_prompt_batch_processing()
    # find_similar_objects_by_example()
    # export_to_coco()
    # custom_export()
    


if __name__ == "__main__":
    main()