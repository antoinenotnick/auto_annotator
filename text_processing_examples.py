"""
Example usage of the sam_segmentation package.

This file demonstrates various ways to use the SAMSegmenter class
and its related utilities.
"""

from pathlib import Path
from sam_segmentation import (
    SAMSegmenter,
    COCOExporter,
    LabelMeExporter,
    OverlayVisualizer,
)

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent


def example_1_simple_batch_processing():
    """Example 1: Simple batch processing with default settings."""
    print("=" * 60)
    print("Example 1: Simple Batch Processing")
    print("=" * 60)

    segmenter = SAMSegmenter(text_prompt="pole")
    results = segmenter.process_directory(SCRIPT_DIR / "images")

    print(f"\nProcessed {len(results)} images")
    for result in results:
        print(f"  - {result.image_path.name}: {result.num_detections} detections")


def example_2_custom_configuration():
    """Example 2: Custom configuration with selective exports."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)

    segmenter = SAMSegmenter(
        text_prompt="person",
        mask_threshold=0.6,           # Higher threshold for more confident detections
        export_format="labelme",      # Only export LabelMe format
        save_overlay=True,
        overlay_alpha=0.5,             # More opaque overlays
        category_name="person",
    )

    results = segmenter.process_directory(SCRIPT_DIR / "images", output_dir=SCRIPT_DIR / "output")
    print(f"\nProcessed {len(results)} images with custom settings")


def example_3_single_image_processing():
    """Example 3: Process a single image."""
    print("\n" + "=" * 60)
    print("Example 3: Single Image Processing")
    print("=" * 60)

    segmenter = SAMSegmenter(text_prompt="car")
    result = segmenter.process_image(SCRIPT_DIR / "images" / "sample.jpg")

    print(f"Image: {result.image_path.name}")
    print(f"Size: {result.image_size[0]}x{result.image_size[1]}")
    print(f"Detections: {result.num_detections}")
    print(f"Scores: {result.scores}")


def example_4_custom_export(): # Segments and exports to COCO json
    """Example 4: Process without exports, then export with custom settings."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Export")
    print("=" * 60)

    # Process without automatic exports (pass empty list)
    segmenter = SAMSegmenter(
        text_prompt="screw",
        export_format=[],  # No automatic exports
        save_overlay=True,
    )

    results = segmenter.process_directory(SCRIPT_DIR / "images")

    # Custom COCO export with different settings
    exporter = COCOExporter(
        category_name="screw",
        dataset_name="screw dataset",
        polygon_tolerance=1.5,  # More precise polygons
    )
    exporter.export(results, SCRIPT_DIR / "output" / "custom_coco.json")
    print("Custom COCO export complete")


def example_5_visualization_only():
    """Example 5: Use visualizer independently."""
    print("\n" + "=" * 60)
    print("Example 5: Visualization Only")
    print("=" * 60)

    from PIL import Image
    import numpy as np

    # Create custom visualizer
    visualizer = OverlayVisualizer(
        alpha=0.6,
        contour_thickness=3,
        draw_labels=True,  # Enable label drawing
    )

    # Assuming you have masks from somewhere
    # image = Image.open("car.jpg")
    # masks = np.load("masks.npy")
    #
    # overlay = visualizer.render(image, masks, labels=["car_1", "car_2", "car_3"])
    # Image.fromarray(overlay).save("overlay.png")

    print("Visualizer can be used independently of the segmenter")


def example_6_export_different_formats():
    """Example 6: Export only specific formats."""
    print("\n" + "=" * 60)
    print("Example 6: Export Different Formats")
    print("=" * 60)

    # LabelMe only
    segmenter_labelme = SAMSegmenter(
        text_prompt="car",
        export_format="labelme",
        save_overlay=False,
    )

    # COCO only (default)
    segmenter_coco = SAMSegmenter(
        text_prompt="car",
        export_format="coco",
        save_overlay=False,
    )

    # Both formats
    segmenter_both = SAMSegmenter(
        text_prompt="car",
        export_format=["labelme", "coco"],
        save_overlay=False,
    )

    # Overlay only
    segmenter_overlay = SAMSegmenter(
        text_prompt="car",
        export_format=[],  # No exports
        save_overlay=True,
    )

    print("Created segmenters for different export formats")


def example_7_custom_output_directories():
    """Example 7: Organize outputs in custom directories."""
    print("\n" + "=" * 60)
    print("Example 7: Custom Output Directories")
    print("=" * 60)

    segmenter = SAMSegmenter(
        text_prompt="car",
        output_dir=SCRIPT_DIR / "results",
        overlay_subdir="visualizations",  # Custom overlay folder name
    )

    results = segmenter.process_directory(SCRIPT_DIR / "images")
    print(f"Results saved to: {SCRIPT_DIR / 'results'}")
    print(f"Overlays saved to: {SCRIPT_DIR / 'results' / 'visualizations'}")


def example_8_process_different_objects():
    """Example 8: Use for different object types."""
    print("\n" + "=" * 60)
    print("Example 8: Process Different Objects")
    print("=" * 60)

    # Segment different objects by changing the prompt
    segmenter_people = SAMSegmenter(
        text_prompt="person",
        category_name="person",
        dataset_name="People Dataset",
    )

    segmenter_animals = SAMSegmenter(
        text_prompt="dog",
        category_name="dog",
        dataset_name="Animal Dataset",
    )

    print("Created segmenters for different object types")


def example_9_export_results_separately():
    """Example 9: Export results separately from processing."""
    print("\n" + "=" * 60)
    print("Example 9: Separate Export")
    print("=" * 60)

    # Process without exports
    segmenter = SAMSegmenter(
        text_prompt="car",
        export_format=[],  # No automatic exports
        save_overlay=False,
    )

    results = segmenter.process_directory(SCRIPT_DIR / "images")

    # Export later with different configurations
    exported = segmenter.export_results(results, SCRIPT_DIR / "exports")

    print(f"Exported to {len(exported)} format(s)")
    for format_name, paths in exported.items():
        print(f"  {format_name}: {paths}")


def example_10_multiple_object_prompts(): # multiple object types in one pass
    """Example 10: Segment multiple object types in one pass."""
    print("\n" + "=" * 60)
    print("Example 10: Multiple Object Prompts")
    print("=" * 60)

    # Segment multiple object types
    segmenter = SAMSegmenter(
        text_prompt=["bus", "car", "bike"],  # Multiple prompts
        category_name="vehicle",
        export_format="coco",
        save_overlay=True,
    )

    results = segmenter.process_directory(SCRIPT_DIR / "traffic_images")

    print(f"\nProcessed {len(results)} images")
    for result in results:
        print(f"  - {result.image_path.name}: {result.num_detections} detections")


def main():
    """Run all examples (commented out to avoid actual execution)."""
    print("SAM Segmentation Package - Usage Examples")
    print("==========================================\n")

    # Uncomment the examples you want to run:

    example_1_simple_batch_processing()
    # example_2_custom_configuration()
    # example_3_single_image_processing()
    # example_4_custom_export()
    # example_5_visualization_only()
    # example_6_export_different_formats()
    # example_7_custom_output_directories()
    # example_8_process_different_objects()
    # example_9_export_results_separately()
    # example_10_multiple_object_prompts()

    print("\nUncomment the examples you want to run in the main() function.")


if __name__ == "__main__":
    main()
