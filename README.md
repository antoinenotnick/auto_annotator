## SAM3 Auto Labeler / Segmentation Toolkit

Configurable object segmentation pipeline built around the SAM‑3 model, with helpers to export results to COCO and LabelMe formats and to generate overlay images for quick visual inspection.

The core functionality lives in the `sam_segmentation` package (`SAMSegmenter`, `SegmentationResult`, `COCOExporter`, `LabelMeExporter`, `OverlayVisualizer`), with small scripts and examples that show how to batch‑process images and export annotations.

---

## Features

- **Reusable segmentation pipeline**: `SAMSegmenter` class for processing single images or whole directories.
- **Batch processing**: Simple helpers to process a folder of images and collect results.
- **Multiple export formats**:
  - COCO JSON (single dataset file)
  - LabelMe JSON (per‑image annotations)
- **Overlay visualization**: Save color overlays and contours for quick QA of segmentations.
- **Examples included**: `examples.py` demonstrates common workflows and configurations.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/antoinenotnick/auto_labeler
cd auto_labeler
```

### 2. Python environment

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

Install the Python requirements:

```bash
pip install -r requirements.txt
```

> **Note**: PyTorch and CUDA are not pinned in `requirements.txt` – install the appropriate `torch`, `torchvision`, and `torchaudio` versions for your system by following the official PyTorch install guide, then come back to this project.

You will also need access to the SAM‑3 weights via Hugging Face; make sure your environment is authenticated if required (e.g. using `huggingface-cli login` or environment variables).

---

## Project Layout

- **`sam_segmentation/`**
  - `segmenter.py`: Implements `SAMSegmenter` and `SegmentationResult`.
  - `exporters.py`: `COCOExporter` and `LabelMeExporter` utilities.
  - `visualizer.py`: `OverlayVisualizer` for generating overlays.
  - `utils.py`: Helper functions like `mask_to_polygon`.
  - `__init__.py`: Public API for the package.
- **`examples.py`**: Script with multiple example workflows (batch processing, custom exports, visualization, etc.).
- **`processing.py`**: Higher‑level workflows and utilities (box prompts, exemplar search, custom COCO export).
- **`sam.py`**: Simple wrapper script that runs a default segmentation pipeline on the `images/` directory.
- **`images/`**: Example images and (optionally) saved outputs.
- **`output/`**: Default target directory for exported annotations and overlays (created automatically as needed).
- **`label_images.py`**: (Experimental) Tkinter/OpenCV GUI skeleton for manual/assisted labeling.

---

## Quick Start

### Run the simple CLI wrapper

From the project root:

```bash
python sam.py
```

By default, this will:

- Look for images in `images/`.
- Run the SAM‑based segmentation using the configured text prompts / labels.
- Save:
  - COCO annotations (e.g. `annotations_coco.json`) into the chosen `output_dir` (often `output/`).
  - Overlay images into an overlay subfolder (e.g. `output/segmented_images/`).

Check `sam.py` if you want to change prompts, labels, or output directories.

---

## Visual box prompts and exemplar‑based search

The project supports **visual (box) prompts** and a simple exemplar‑based search workflow via `processing.py`.

- **Configure the object and box** in `processing.py`:

```python
OBJECT = "pole"  # label / text prompt for your object
box_prompt = [x, y, w, h]  # box in pixels, from the top‑left corner
```

- **Box‑prompt batch processing**:

```python
from processing import box_prompt_batch_processing

box_prompt_batch_processing()
```

This uses only the **visual box prompt** (no text) across the `images/` directory and writes:

- An `annotations_coco.json` file into `output/`.
- Overlay images (with masks and the cyan prompt box drawn on the reference image) into `output/segmented_images/`.

- **Exemplar‑based search** (mask from one image, then “find similar” in others):

```python
from pathlib import Path
from processing import find_similar_objects_by_example

matches = find_similar_objects_by_example(
    ref_image=Path("images/1pole-down.jpg"),  # image where you drew the box_prompt
    hist_threshold=0.7,                       # loosen/tighten similarity
)
```

This does:

- Runs SAM‑3 with the **box prompt** on `ref_image` to obtain a reference mask.
- Computes a simple appearance descriptor (color histogram) from that mask.
- Runs SAM‑3 with the **text prompt** `OBJECT` on all images and keeps only detections whose appearance is sufficiently similar to the reference.
- Returns a list of `(SegmentationResult, similarity)` for images that contain at least one similar object and saves overlays/COCO into `output/`.

---

## Using the `SAMSegmenter` in Your Own Code

Basic usage from Python:

```python
from sam_segmentation import SAMSegmenter
from pathlib import Path

segmenter = SAMSegmenter(
    text_prompt="car",          # or list of prompts
    export_format=["coco"],     # "coco", "labelme", [], or a list
    save_overlay=True,          # whether to save overlay images
)

results = segmenter.process_directory(Path("images"))
print(f"Processed {len(results)} images")
for r in results:
    print(r.image_path.name, r.num_detections)
```

Each entry in `results` is a `SegmentationResult` instance that contains:

- `image_path` and `image_size`
- `masks` (tensor or NumPy array)
- `boxes` / `scores` (if provided by the model)
- `categories` / labels (if configured)

---

## Exporting to COCO and LabelMe

You can either let `SAMSegmenter` export automatically by setting `export_format`, or export manually using the exporters.

### Automatic exports via `SAMSegmenter`

```python
segmenter = SAMSegmenter(
    text_prompt="person",
    export_format=["coco", "labelme"],
    save_overlay=True,
    output_dir=Path("output"),
)

results = segmenter.process_directory(Path("images"))
```

This will:

- Write a single COCO JSON file to `output/`.
- Write per‑image LabelMe JSON files (alongside images or into the chosen output directory).
- Save overlays (by default under a subdirectory like `output/segmented_images/`).

### Manual COCO export

`examples.py` shows how to first process without any automatic exports, then write a custom COCO JSON:

```python
from pathlib import Path
from sam_segmentation import SAMSegmenter, COCOExporter

script_dir = Path(__file__).parent

segmenter = SAMSegmenter(
    text_prompt="pole",
    export_format=[],           # no automatic exports
    save_overlay=False,
)

results = segmenter.process_directory(script_dir / "images")

exporter = COCOExporter(
    category_name="pole",
    dataset_name="pole dataset",
    polygon_tolerance=1.5,
)

exporter.export(results, script_dir / "output" / "custom_coco.json")
```

The exporter will create the parent directory of the output path if it does not exist.

---

## Examples Script

`examples.py` contains a collection of small, focused examples:

- **Example 1** – Simple batch processing with default settings.
- **Example 2** – Custom configuration with selective exports and overlays.
- **Example 3** – Single image processing.
- **Example 4** – Custom export: process once, export to COCO later.
- **Example 5** – Use `OverlayVisualizer` independently.
- **Example 6** – Export only specific formats.
- **Example 7** – Custom output directories.
- **Example 8** – Process different object types by changing prompts.
- **Example 9** – Export results separately via `segmenter.export_results`.
- **Example 10** – Multiple object prompts in one pass.

To run a particular example, edit `main()` in `examples.py` and uncomment the example call you want.

```bash
python examples.py
```

---

## Notes and Tips

- **Model weights / SAM‑3**: This project expects a SAM‑3 model available through the Hugging Face ecosystem. Check `segmenter.py` for how the model is loaded and confirm you have the right env variables / tokens configured.
- **Performance**: Large images or many masks per image can be memory‑intensive; consider resizing images or adjusting thresholds (`mask_threshold`, `polygon_tolerance`) to trade accuracy for speed.
- **Coordinate systems**: COCO exports use pixel coordinates in the standard COCO format (xywh). Polygons are generated from masks using `mask_to_polygon`.

