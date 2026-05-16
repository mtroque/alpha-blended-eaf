"""Convert RDD2022 Japan PASCAL VOC annotations to YOLOv8 format.

This script is unchanged in spirit from the user's original xml_parser.py:
it walks the train/annotations/xmls directory, parses each XML, converts
bounding boxes to normalized YOLO coordinates, and writes one .txt per image
into train/labels/.

Differences from the original:
  - CLI arguments instead of hardcoded paths, so it can be invoked from the
    bash entry-point script with different roots.
  - Stricter validation (warnings for skipped classes, malformed boxes).
"""

import argparse
import glob
import os
import xml.etree.ElementTree as ET


# Class mapping from RDD2022 Japan damage labels to integer class IDs.
# Order matches configs/dataset.yaml exactly.
CLASS_MAPPING = {
    "D00": 0,  # Longitudinal wheel mark
    "D01": 1,  # Longitudinal construction joint
    "D10": 2,  # Lateral equal interval
    "D11": 3,  # Lateral construction joint
    "D20": 4,  # Alligator crack
    "D40": 5,  # Pothole/rutting
    "D43": 6,  # Crosswalk blur
    "D44": 7,  # White line blur
    "D50": 8,  # Manhole
}


def voc_to_yolo(size, box):
    """Convert (xmin, xmax, ymin, ymax) in pixels to YOLO normalized format."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    width = box[1] - box[0]
    height = box[3] - box[2]
    return (x_center * dw, y_center * dh, width * dw, height * dh)


def process(xml_dir, labels_out_dir):
    os.makedirs(labels_out_dir, exist_ok=True)
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))

    if not xml_files:
        print(f"ERROR: no XML files in {xml_dir}")
        return 1

    print(f"Processing {len(xml_files)} XML files...")
    converted = 0
    skipped_classes = set()
    skipped_boxes = 0

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            size_el = root.find("size")
            w = int(size_el.find("width").text)
            h = int(size_el.find("height").text)

            yolo_lines = []
            for obj in root.findall("object"):
                cls = obj.find("name").text
                if cls not in CLASS_MAPPING:
                    skipped_classes.add(cls)
                    continue

                bb = obj.find("bndbox")
                xmin = float(bb.find("xmin").text)
                xmax = float(bb.find("xmax").text)
                ymin = float(bb.find("ymin").text)
                ymax = float(bb.find("ymax").text)

                if xmin >= xmax or ymin >= ymax:
                    skipped_boxes += 1
                    continue

                ybox = voc_to_yolo((w, h), (xmin, xmax, ymin, ymax))
                yolo_lines.append(
                    f"{CLASS_MAPPING[cls]} {ybox[0]:.6f} {ybox[1]:.6f} {ybox[2]:.6f} {ybox[3]:.6f}"
                )

            base = os.path.splitext(os.path.basename(xml_file))[0]
            out_path = os.path.join(labels_out_dir, f"{base}.txt")
            with open(out_path, "w") as f:
                if yolo_lines:
                    f.write("\n".join(yolo_lines))
            converted += 1

        except Exception as e:
            print(f"WARN: failed to process {xml_file}: {e}")

    print(f"\nProcessed: {converted}/{len(xml_files)} files")
    if skipped_classes:
        print(f"Skipped classes (not in mapping): {sorted(skipped_classes)}")
    if skipped_boxes:
        print(f"Skipped malformed boxes: {skipped_boxes}")
    print(f"Labels written to: {labels_out_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Convert RDD2022 VOC XML to YOLO format.")
    parser.add_argument(
        "--xml-dir",
        required=True,
        help="Directory containing the XML annotation files "
             "(e.g. /content/src/Japan/train/annotations/xmls)",
    )
    parser.add_argument(
        "--labels-out-dir",
        required=True,
        help="Directory to write YOLO .txt label files "
             "(e.g. /content/src/Japan/train/labels)",
    )
    args = parser.parse_args()
    return process(args.xml_dir, args.labels_out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
