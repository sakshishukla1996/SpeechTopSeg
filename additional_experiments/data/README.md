# Data

## Audio

To download audio files, make sure to install [yt-dlp](https://pypi.org/project/yt-dlp/)
and [ffmpeg](https://www.ffmpeg.org/), and run the following command:

```shell
python download_audio.py
```

This might take several hours to finish.

Audio files will be located in `wav` subdirectory of each directory.

## Partitions

List files in each directory correspond to the data partitions:
 - `train.lst` -- traininig data;
 - `val.lst` -- validation (development) data;
 - `test.lst` -- testing (evaluation) data.

Training and validation data are not provided for the `akashvani` subset.

## JSON

JSON files are located in `json` subdirectory of each directory.

All JSON files contain two fields:
 - `chapters` -- topic segments;
 - `sentences` -- automatic transcriptions divided into sentences and grouped into topic segments.

Transcriptions are not provided for the `akashvani` subset
and for the `euronews_en/json/WY66vzBSWxg.json` and `euronews_de/json/PzcL3Gjvxcg.json` files.
