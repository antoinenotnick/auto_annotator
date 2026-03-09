from pathlib import Path

import cv2
import numpy as np

from sam_segmentation import SAMSegmenter, COCOExporter
from sam_segmentation.utils import load_image_with_exif

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent

# Image to process for visual or text‑prompt examples
IMG_PATH = SCRIPT_DIR / "images" / "<IMAGE NAME HERE>.jpg"

# Object label / text prompt
OBJECT = "<OBJECT NAME HERE>"

# Fallback visual prompt: (x, y, w, h) in pixels from top‑left corner
box_prompt = [0.0, 0.0, 0.0, 0.0]


def select_box_prompt(image_path: Path) -> list[float] | None:
    """
    Open an interactive window to draw a visual box prompt on an image.

    The user drags a rectangle with the mouse; on confirmation, the box
    is returned in pixel coordinates as `[x, y, w, h]`.

    Args:
        image_path: Path to the image where the box will be drawn.

    Returns:
        A list `[x, y, w, h]` in original image pixels, or `None` if the
        user cancels (ESC or q).
    """
    # Load with EXIF handling, then convert to OpenCV BGR
    pil_img = load_image_with_exif(image_path, enable_exif=True)
    img_rgb = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    clone = img_bgr.copy()
    drawing = {"start": None, "end": None, "dragging": False}
    window_name = "Draw visual box prompt (ENTER=accept, ESC=cancel)"

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing["start"] = (x, y)
            drawing["end"] = (x, y)
            drawing["dragging"] = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing["dragging"]:
            drawing["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing["dragging"]:
            drawing["end"] = (x, y)
            drawing["dragging"] = False

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        frame = clone.copy()
        if drawing["start"] and drawing["end"]:
            x0, y0 = drawing["start"]
            x1, y1 = drawing["end"]
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(20) & 0xFF

        # ENTER or Space to accept
        if key in (13, 32):  # Enter or Space
            if drawing["start"] and drawing["end"]:
                x0, y0 = drawing["start"]
                x1, y1 = drawing["end"]
                x_min, x_max = sorted([x0, x1])
                y_min, y_max = sorted([y0, y1])
                w = x_max - x_min
                h = y_max - y_min
                cv2.destroyWindow(window_name)
                if w > 0 and h > 0:
                    return [float(x_min), float(y_min), float(w), float(h)]
            # If no valid box, keep waiting

        # ESC or q to cancel
        if key in (27, ord("q")):
            cv2.destroyWindow(window_name)
            return None

def single_image_processing_with_text_prompt():
    """
    Process a single image using only a text prompt.

    Uses `OBJECT` as the text prompt and `IMG_PATH` as the input image,
    saving outputs to `output/`.
    """
    print("\n" + "=" * 60)
    print("Single Image Processing")
    print("=" * 60)

    segmenter = SAMSegmenter(
        text_prompt=OBJECT,
        category_name=OBJECT,
    )

    result = segmenter.process_image(
        IMG_PATH,
        output_dir=SCRIPT_DIR / "output",
    )

    print(f"Image: {result.image_path.name}")
    print(f"Size: {result.image_size[0]}x{result.image_size[1]}")
    print(f"Detections: {result.num_detections}")
    print(f"Scores: {result.scores}")

    return result


def single_image_processing_with_box_prompt():
    """
    Process a single image using an interactive visual box prompt.

    The user draws a box on `IMG_PATH`; that box is passed as a visual
    prompt to SAM3 for that image only.
    """
    print("\n" + "=" * 60)
    print("Single Image Processing")
    print("=" * 60)

    # Let the user draw a visual box prompt; fall back to the default if cancelled
    user_box = select_box_prompt(IMG_PATH)
    current_box = user_box if user_box is not None else box_prompt
    if user_box is not None:
        print(f"Using user‑selected box_prompt: {current_box}")
    else:
        print(f"No box selected; using fallback box_prompt: {current_box}")

    segmenter = SAMSegmenter(
        text_prompt=None,
        box_prompt=current_box,
        category_name=OBJECT,
    )

    result = segmenter.process_image(
        IMG_PATH,
        output_dir=SCRIPT_DIR / "output",
    )

    print(f"Image: {result.image_path.name}")
    print(f"Size: {result.image_size[0]}x{result.image_size[1]}")
    print(f"Detections: {result.num_detections}")
    print(f"Scores: {result.scores}")

    return result


def text_prompt_batch_processing():
    """
    Batch‑process all images in `SCRIPT_DIR / "images"` using only a text prompt.

    Uses `OBJECT` as the text prompt and writes outputs into `output/`.
    """
    print("=" * 60)
    print("Text Prompt Batch Processing")
    print("=" * 60)

    segmenter = SAMSegmenter(
        text_prompt=OBJECT,
        category_name=OBJECT,
        save_overlay=True,
    )

    results = segmenter.process_directory(
        SCRIPT_DIR / "images",
        output_dir=SCRIPT_DIR / "output",
    )

    print(f"\nProcessed {len(results)} images")
    for res in results:
        print(f"  - {res.image_path.name}: {res.num_detections} detections")

    return results 


def box_prompt_batch_processing():
    """
    Batch processing using a visual exemplar selected on the reference image.

    Steps:
      1) Show the reference image (`IMG_PATH`) and let the user draw a box.
      2) Use that box as a visual prompt to get a mask on the reference image.
      3) Run text‑prompt segmentation on all images and keep only objects that
         look similar to the reference mask (via color‑histogram similarity).
    """
    return find_similar_objects_by_example(ref_image=IMG_PATH)


def _compute_mask_histogram(image_array: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """
    Compute a color histogram for the region defined by a binary mask.

    Args:
        image_array: RGB image as a NumPy array of shape (H, W, 3).
        mask: Mask array (any shape broadcastable to (H, W)); values > 0.5
              are treated as foreground.

    Returns:
        A 1D normalized histogram vector, or None if the mask is empty.
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
    2‑stage exemplar workflow:
      1) Use a box prompt on a reference image to get a mask for the example object.
      2) Use a text prompt on all images and keep only detections whose appearance
         (color histogram) is similar to the example mask.

    Args:
        ref_image: Reference image containing the object you draw the box on.
                   Defaults to `IMG_PATH`.
        hist_threshold: Similarity threshold (OpenCV CORREL) in [‑1, 1]; higher = stricter.

    Returns:
        List of (SegmentationResult, best_similarity) for images that contain at
        least one similar object.
    """
    if ref_image is None:
        ref_image = IMG_PATH

    # 1) Let the user choose a visual box on the reference image (or fall back)
    user_box = select_box_prompt(ref_image)
    current_box = user_box if user_box is not None else box_prompt
    if user_box is not None:
        print(f"Using user‑selected box_prompt for exemplar: {current_box}")
    else:
        print(f"No box selected on exemplar; using fallback box_prompt: {current_box}")

    # 2) Get reference mask from visual (box) prompt on the reference image
    ref_segmenter = SAMSegmenter(
        text_prompt=None,
        box_prompt=current_box,
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
    """
    Export a list of segmentation results to a COCO JSON file.

    Args:
        results: List of `SegmentationResult` objects to export.
    """
    exporter = COCOExporter(
        category_name=OBJECT,
        dataset_name="My Dataset",
    )
    exporter.export(results, "annotations.json")


def custom_export():
    """
    Process a directory without automatic exports, then export with custom COCO settings.

    This shows how to decouple segmentation from export configuration.
    """
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
    """
    Entry point for running processing examples.

    Uncomment exactly one workflow below to run it.
    """

    single_image_processing_with_text_prompt()
    # single_image_processing_with_box_prompt()
    # text_prompt_batch_processing()
    # box_prompt_batch_processing()
    # find_similar_objects_by_example()
    # export_to_coco(text_prompt_batch_processing())
    # custom_export()


if __name__ == "__main__":
    main()