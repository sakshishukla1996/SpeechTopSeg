# Segmentation purity and coverage F-score

Two similar scripts are provided to calculate the segmentation purity and coverage F-score
([metrics explanation](https://pyannote.github.io/pyannote-metrics/reference.html#segmentation)):
 - `segmentation_f_measure_e2e.py` -- intended for the output of an audio-based end-to-end system, which provides one label for each window of a fixed size (currently hardcoded to 10 seconds);
 - `segmentation_f_measure_pipeline.py` -- intended for the output of a text-based pipeline system, which provides one label for each sentence.

Usage is the same for both scripts:

```shell
python segmentation_f_measure_e2e.py /path/to/reference /path/to/hypothesis
```
Here `/path/to/reference` is the path to the reference data, for example `../data/euronews_en/`,
and `/path/to/hypothesis` is the path to the system output.
