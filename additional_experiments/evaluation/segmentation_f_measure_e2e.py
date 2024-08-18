import glob
import json
import sys
import torch

from pyannote.core import Timeline, Segment, Annotation
from pyannote.metrics.segmentation import SegmentationPurityCoverageFMeasure

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} /path/to/reference /path/to/hypothesis")
    sys.exit(1)

ref_dir = sys.argv[1]
hyp_dir = sys.argv[2]

scores = []
f_measure = SegmentationPurityCoverageFMeasure()

for pt in glob.glob(f"{hyp_dir}/*pt"):
    video = pt[-26:-15]
    ref_info = json.load(open(f"{ref_dir}/json/{video}.json"))

    ref = Annotation()

    for chapter in ref_info["chapters"]:
        start = int(chapter["start_time"])
        end = int(chapter["end_time"])
        ref[Segment(start, end)] = "t"

    hyp = Annotation()

    start, end = 0, 0

    preds = torch.load(pt, map_location="cpu")
    timestamps = (
        torch.flatten(preds["logits"].argmax(-1).squeeze().nonzero()) + 1
    ) * 10

    for timestamp in timestamps.tolist():
        end = timestamp
        hyp[Segment(start, end)] = "t"
        start = end

    if len(hyp) == 0:
        score = 0.0
    else:
        score = f_measure(ref, hyp)
    scores.append(score)

print(f"{hyp_dir} {torch.tensor(scores).mean().item()}")
