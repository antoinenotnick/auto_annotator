from pathlib import Path

import cv2
import numpy as np

from sam_segmentation import SAMSegmenter, COCOExporter
from sam_segmentation.visualizer import OverlayVisualizer
from sam_segmentation.utils import load_image_with_exif

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent

# Image to process for specific visual or text‑prompt examples
IMG_PATH = SCRIPT_DIR / "images" / "pole_ex2.jpg"

# Object label / text prompt
OBJECT = "insulator"

# Fallback visual prompt: (x, y, w, h) in pixels from top‑left corner
box_prompt = [0.0, 0.0, 0.0, 0.0]


def select_visual_box_prompt(image_path: Path) -> list[float] | None:
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

def export_to_coco(results):
    """
    Export a list of segmentation results to a COCO JSON file.

    Args:
        results: List of `SegmentationResult` objects to export.
    """
    
    print("\n" + "=" * 60)
    print("Export to COCO")
    print("=" * 60)

    if results is None:
        return

    if not isinstance(results, list):
        normalized_results = [results]
    else:
        normalized_results = results

    if len(normalized_results) == 0:
        print("No results to export to COCO.")
        return

    exporter = COCOExporter(
        category_name=OBJECT,
        dataset_name="My Dataset",
    )

    output_path = SCRIPT_DIR / "output" / "annotations.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exporter.export(normalized_results, output_path)


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


def text_prompt_single_image_processing():
    """
    Process a single image using only a text prompt.

    Uses `OBJECT` as the text prompt and `IMG_PATH` as the input image,
    saving outputs to `output/`.
    """
    print("\n" + "=" * 60)
    print("Single Image Processing with Text Prompt")
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

    export_to_coco(result)

    return result


def visual_prompt_single_image_processing():
    """
    Process a single image using an interactive visual box prompt.

    The user draws a box on `IMG_PATH`; that box is passed as a visual
    prompt to SAM3 for that image only.
    """
    print("\n" + "=" * 60)
    print("Single Image Processing with Visual Prompt")
    print("=" * 60)

    # Let the user draw a visual box prompt; fall back to the default if cancelled
    user_box = select_visual_box_prompt(IMG_PATH)
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

    export_to_coco(result)

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

    export_to_coco(results)

    return results 


def visual_prompt_batch_processing():
    """
    Batch-process all images by drawing a separate visual box prompt per image.

    For each image in `SCRIPT_DIR / "images"`:
      1) Show the image in an interactive window.
      2) Let the user draw a box (or skip with ESC/q).
      3) Run `SAMSegmenter` once for that image using only the box prompt.
      4) Save overlays (with masks; the box is only used for prompting, not required in overlays).
    """

    print("\n" + "=" * 60)
    print("Visual Prompt Batch Processing")
    print("=" * 60)

    images_dir = SCRIPT_DIR / "images"
    image_paths = sorted(p for p in images_dir.iterdir() if p.is_file())
    if not image_paths:
        print(f"No images found in {images_dir}")
        return []

    results = []
    for idx, image_path in enumerate(image_paths, start=1):
        print(f"\n[{idx}/{len(image_paths)}] Selecting box for {image_path.name}")
        user_box = select_visual_box_prompt(image_path)
        current_box = user_box if user_box is not None else box_prompt
        if user_box is not None:
            print(f"Using user-selected box_prompt: {current_box}")
        else:
            print(f"No box selected; using fallback box_prompt: {current_box}")

        segmenter = SAMSegmenter(
            text_prompt=None,
            box_prompt=current_box,
            category_name=OBJECT or "object",
            save_overlay=True,
        )
        result = segmenter.process_image(
            image_path,
            output_dir=SCRIPT_DIR / "output",
        )
        print(
            f"  -> {result.image_path.name}: "
            f"{result.num_detections} detections"
        )
        results.append(result)

    export_to_coco(results)
    return results


def main():
    """
    Entry point for running processing examples.

    Uncomment exactly one workflow below to run it.
    """

    # Missing some export to coco functions that could be a pain for users

    # text_prompt_single_image_processing()
    # visual_prompt_single_image_processing()
    # text_prompt_batch_processing()
    # visual_prompt_batch_processing()
    text_prompt_batch_processing()
    # custom_export()


if __name__ == "__main__":
    main()