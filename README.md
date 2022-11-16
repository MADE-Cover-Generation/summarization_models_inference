## Description

This repository is intended for generating video summaries using 4 cutting edge summarization models: PGL-SUM, CA-SUM, DSNet anchor based and DSNet anchor free. Models are pretrained on TVSum and SumMe datasets.

## Installation
Create virtual environment:

`python -m venv .summarization`

Activate virtual environment:

`source .summarization/bin/activate`

Install dependencies:

`pip install -r requirements.txt`

## Usage

Summary generation:

First move to the source folder:

`cd src`

Single video:

`python inference.py pglsum --source ../custom_data/videos/source_video_name.mp4 --save-path ./output/summary_video_name.mp4 --sample-rate 30 --final-frame-length 30` 

Folder of videos:

`python inference.py pglsum --source ../custom_data/videos/source_video_folder --save-path ./output/summary_videos_folder --sample-rate 30 --final-frame-length 30`

Eligible model names: `pglsum` - PGL-SUM, `casum` - CA-SUM, `dsnet_ab` - DSNet anchor based and `dsnet_af` - DSNet anchor free.

`--sample-rate 30` means the model will take every 30th frame for analysis 

`--final-frame-length 30` means the final video summary will have 30 frames (around 30 sec video)



