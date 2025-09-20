# Hierarchical Reward Modeling for Fault Localization

This repository contains the official implementation of **HiFL**, a hierarchical fault localization method presented in our paper.

## Training Dataset: HiFL-44k

The reward model was trained on the **HiFL-44k** dataset, which contains approximately 44,000 training samples.

You can access the dataset on Hugging Face:
[https://huggingface.co/datasets/lapsel/HiFL-44k](https://huggingface.co/datasets/lapsel/HiFL-44k)

## Setup

### Environment Variables
Before running the scripts, configure the following environment variables.

```bash
# Your OpenAI API key
export OPENAI_API_KEY=''

# Add the project directory to your Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Path to the directory containing repository structures
export PROJECT_FILE_LOC="/path/to/repo_structures"
```

## Usage Workflow

The fault localization process is divided into three main stages: File-Level, Function-Level, and Line-Level localization. Follow the steps below in order.

### Stage 1: File-Level Localization

This stage identifies the most relevant files for a given bug.

#### 1. Localize Relevant Files
This command identifies files that are likely related to the bug.

```shell
python agentless/fl/localize.py \
    --model model_name \
    --file_level \
    --output_folder results/.../file_level \
    --num_threads 10 \
    --sample 3 \
    --skip_existing \
    --pred_list .../file_level/loc_outputs.jsonl
```

#### 2. Localize Irrelevant Files
This command identifies files that are likely *not* related to the bug, helping to narrow the search space.

```shell
python agentless/fl/localize.py \
    --model model_name \
    --file_level \
    --irrelevant \
    --output_folder results/.../file_level_irrelevant \
    --num_threads 15 \
    --sample 3 \
    --skip_existing \
    --pred_list .../file_level_irrelevant/loc_outputs.jsonl
```

#### 3. Retrieve by Similarity
Next, use similarity-based retrieval to find related code snippets from the localized irrelevant files.

```shell
python agentless/fl/retrieve.py \
    --index_type simple \
    --filter_type given_files \
    --filter_file results/swe-bench-lite/file_level_irrelevant/loc_outputs.jsonl \
    --output_folder results/swe-bench-lite/retrievel_embedding \
    --persist_dir embedding/swe-bench_simple \
    --num_threads 10
```

#### 4. Merge File Lists
Finally, merge the results from the model-based and retrieval-based localization.

```shell
python agentless/fl/combine.py \
    --retrieval_loc_file results/swe-bench-lite/retrievel_embedding/retrieve_locs.jsonl \
    --model_loc_file results/swe-bench-lite/file_level/loc_outputs.jsonl \
    --top_n 3 \
    --output_folder results/swe-bench-lite/file_level_combined
```

### Stage 2: Function-Level Localization

Using the file list from the previous stage, this step pinpoints relevant functions and code elements.

```shell
python agentless/fl/localize.py \
    --related_level \
    --model model_name \
    --output_folder results/.../related_elements \
    --top_n 3 \
    --compress_assign \
    --compress \
    --start_file .../file_level_combined/combined_locs.jsonl \
    --num_threads 15 \
    --skip_existing \
    --sample 3 \
    --pred_list .../related_elements/loc_outputs.jsonl
```

### Stage 3: Line-Level Localization

This final stage identifies the exact lines of code that need to be edited to fix the bug.

```shell
python agentless/fl/localize.py \
    --fine_grain_line_level \
    --model model_name \
    --output_folder results/.../edit_location_samples \
    --top_n 3 \
    --compress \
    --temperature 1 \
    --num_samples 1 \
    --start_file .../related_elements/loc_outputs.jsonl \
    --num_threads 12 \
    --skip_existing \
    --sample 3 \
    --pred_list .../edit_location_samples/loc_outputs.jsonl
```