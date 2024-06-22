from pathlib import Path
import torchaudio
import torch
from tqdm import tqdm


language = "es"
# input_root = Path(f"/data/tagesschau/audio_dataset/train_data/")
# input_root = Path(f"/data/tagesschau/audio_dataset/val_data/")
input_root = Path(f"/data/akashvani/all_all_all_predictions/")
# files = list(input_root.glob("**/*.wav"))
files = list(input_root.glob("*.pt"))
json_files = list([i for i in Path("/data/akashvani/prob_labels/").glob("*.json")])

durations = []
segments = []

for i, file in enumerate(json_files):
    # if str(file.stem).split("_")[0] not in json_files:
    #     continue
    # meta = torchaudio.info(file)
    # duration = meta.num_frames / meta.sample_rate
    req = (str(file.stem).split("_")[0] + ".wav")
    file = Path("/data/akashvani/nur_wav") / req
    if not file.exists():
        continue
    a = torchaudio.info(file)
    # duration = (a['timestamps'][-1] / 60).item()
    duration = a.num_frames/a.sample_rate
    print(i, file, duration)
    durations.append(duration)
    # segments.append(a['labels'].sum().item())

total_durations_s = sum(durations)
total_duration_mins = total_durations_s / 60
total_duration_hours = total_duration_mins / 60

avg_duration_s = sum(durations) / len(durations)
avg_duration_min = avg_duration_s / 60
avg_duration_hours = avg_duration_min / 60
breakpoint()

print(f"Found {len(durations)} in {input_root}")
print(f"Total duration = {total_durations_s}s {total_duration_mins}min {total_duration_hours}:{total_duration_hours%60} hr")
print(f"Average duration = {avg_duration_s}s {avg_duration_min}min {avg_duration_hours} hr")

# print(sum(segments) / len(segments))