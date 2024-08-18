import glob
import json
import segeval
import torch
import sys

from nltk.tokenize.texttiling import TextTilingTokenizer

from pyannote.core import Timeline, Segment, Annotation
from pyannote.metrics.segmentation import SegmentationPurityCoverageFMeasure

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} /path/to/data")
    sys.exit(1)

idir = sys.argv[1]
test_videos = [x.strip() for x in open(f"{idir}/test.lst")]

docs = {"test": []}
labels = {"test": []}
refs = {"test": []}

for video in test_videos:
    info = json.load(open(f"{idir}/json/{video}.json"))

    docs[dset].append([x for s in info["sentences"] for x in s])
    labels[dset].append([[0] * (len(s) - 1) + [1] for s in info["sentences"]])

    ref = Annotation()

    for chapter in info["chapters"]:
        start = int(chapter["start_time"])
        end = int(chapter["end_time"])
        ref[Segment(start, end)] = "t"

    refs[dset].append(ref)

f_measure = SegmentationPurityCoverageFMeasure()

f_scores = []
p_ks = []
window_diffs = []

tt = TextTilingTokenizer()

for d, doc in enumerate(docs["test"]):
    char2sentence = sum([[i] * (len(s["sentence"]) + 2) for i, s in enumerate(doc)], [])
    text = "\n\n".join([s["sentence"] for s in doc])
    tokenized_text = tt.tokenize(text)

    predictions = [0 for _ in range(len(labels["test"][d]))]
    predictions[-1] = 1

    hyp = Annotation()

    start, end = 0, 0
    char = 0

    for topic in tokenized_text:
        char += len(topic)

        predictions[char2sentence[char]] = 1

        end = int(doc[char2sentence[char]]["timestamp"][1])
        hyp[Segment(start, end)] = "t"
        start = end

    score = f_measure(refs["test"][d], hyp)
    f_scores.append(score)

    ref_masses = segeval.convert_nltk_to_masses(
        "".join([str(x) for x in labels["test"][d]])
    )
    hyp_masses = segeval.convert_nltk_to_masses("".join([str(x) for x in predictions]))

    p_ks.append(float(segeval.pk(hyp_masses, ref_masses)))
    window_diffs.append(float(segeval.window_diff(hyp_masses, ref_masses)))

print(f"P_k {torch.tensor(p_ks).mean().item()}")
print(f"Win_diff {torch.tensor(window_diffs).mean().item()}")
print(f"Segmentation F-Measure {torch.tensor(f_scores).mean().item()}")
