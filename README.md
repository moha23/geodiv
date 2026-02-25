## GeoDiv: Framework for Measuring Geographical Diversity in Text-to-Image Models (ICLR 2026) 

[Abhipsa Basu*](https://abhipsabasu.github.io/), [Mohana Singh*](https://github.com/moha23), [Shashank Agnihotri](https://github.com/shashankskagnihotri),
[Margret Keuper](https://www.uni-mannheim.de/dws/people/professors/prof-dr-ing-margret-keuper/), [R. Venkatesh Babu](https://cds.iisc.ac.in/faculty/venky/)

<sup><small>*Equal contribution</small></sup>

This is the official repository for **GeoDiv: Framework for Measuring Geographical Diversity in Text-to-Image Models** (ICLR 2026).

<p align="center">
    <a href="https://arxiv.org/abs/">Paper</a> | <a href="https://abhipsabasu.github.io/geodiv/">Project Page</a>
</p>


## Overview

**The Problem**: Text-to-image (T2I) models often produce limited and stereotyped visual variation for the same prompt across different countries (e.g., “a photo of a house in Nigeria”), failing to reflect the rich diversity observed in real-world imagery.

<p align="center">
    <img src="assets/problem.svg" width="90%" alt="Problem">
    <br>
    <em>Figure 1: Lack of Geographical Diversity observed in T2I Generations</em>
</p>

GeoDiv provides an automated, reference-free framework that can quantify such fine-grained geographical differences by evaluating images along two complementary axes. 
    
* **SEVI (Socio-Economic Visual Index)**: Captures socio-economic cues through two interpretable dimensions, (a) *Affluence*, ranging from impoverished to affluent depictions, and (b) *Maintenance*, measuring physical condition from worn to pristine.
    
* **VDI (Visual Diversity Index)**: Measures variation in (a) *Entity Appearance*, reflecting attributes such as shape, material, or color of the primary entity, and (b) *Background Appearance*, capturing contextual variability (e.g., type of roads visible).
    
<p align="center">
    <img src="assets/geodiv.svg" width="90%" alt="Axes">
    <br>
    <em>Figure 2: GeoDiv provides an automated, reference-free framework that can quantify such fine-grained geographical differences by evaluating images along four interpretable axes: Entity-Appearance (sloped/flat roof), Background-Appearance (paved/unpaved road), Affluence (luxury/modest settings), and Maintenance (manicured/unkempt). Examples show how the same entity type varies dramatically across countries and generative models.</em>
</p>
     
## Table of Contents

* [Overview](#overview)
* [Setup](#setup)
    * [Environment Setup Instructions](#environment-setup-instructions)
    * [Data Preparation](#data-preparation)
* [Usage](#usage)
    * [Running the Full Pipeline](#running-the-full-pipeline)
    * [Pipeline Components](#pipeline-components)
    * [Running Individual Components](#running-individual-components)
    * [Collating Results](#collating-results)
* [Released Resources](#released-resources)
    * [Question-Answer Set](#question-answer-set)
    * [Prompts](#prompts)
    * [Human Annotations](#human-annotations)
    * [Results](#results)
* [Acknowledgements](#acknowledgements)
* [Citation](#cite-this-work)


## Setup

### Environment Setup Instructions

* Conda environment installation for using Google Cloud SDK:
    ```
    conda create -n geodiv python=3.10.0
    conda activate geodiv
    pip install -r requirements.txt
    ```
* If you want to use Qwen instead:
    ```
    conda create -n geodiv-qwen python=3.10.18
    conda activate geodiv-qwen
    pip install -r requirements_qwen.txt
    ```

* Google Cloud Setup (Required for Gemini Flash)

    * Create or use an existing Google Cloud project (PROJECT_ID). 
    * Install and initialize the Google Cloud SDK: `gcloud init` 
    Installation guide: https://docs.cloud.google.com/sdk/docs/install-sdk#linux 
    * Upload the `images/` folder to the cloud environment (see `scripts/upload_gcloud.sh` for helper code and [Data Preparation](#data-preparation) for image generation instructions).
    * Optional: If using batch-processing (time and cost-efficient):
        * Create a bucket where results will be stored 

            ```
            gcloud storage buckets create gs://geodiv-batch-results \
                --project=PROJECT_ID \
                --default-storage-class=COLDLINE \
                --location=us-central1 \
                --uniform-bucket-level-access \
                --enable-hierarchical-namespace
            ```

        * Grant the Vertex AI service account write access to this results bucket to enable automatic result storage.

### Data Preparation

The following four CSV files and one image folder must be prepared before running the pipeline.

All other files and folders are generated automatically during execution.

1. `country_prompts.csv` :  Contains the country-entity prompts used for image generation: 
`"A photo of a <entity> in <country>"`
You may create this file or use the one provided in the `data` folder. Required columns: [prompt_id, prompt, entity, entity_id, region]. 

2. `all.csv` : Contains all the question and answer pairs used to analyse the entity-appearance dimension of VDI, merged with country and entity combinations. Each question has a `flag` column with values:

    * F (Fixed):
        Determinate type, very few answer choices, typically one correct answer per image.

    * NF (Not Fixed):
        Larger answer lists, multiple answers may be correct simultaneously.

3. `visibility.csv` : For each question flagged NF in `all.csv`, this file contains a corresponding visibility-check question to verify whether the inspected attribute is visible in the image.

    This helps mitigate hallucination-related errors in VQA responses.

4. `step1.csv` : Contains one question per country and entity combination, to verify if any background element is visible in the image. Subsequent background-analysis steps are generated automatically based on VQA outputs from previous steps.

5. `images/` : For each prompt in `country_prompts.csv`, generate a fixed number of images per prompt using different text-to-image (T2I) models. We use 250 images per prompt in the paper; however, we find 100 images sufficient for stable estimates.

Directory structure:

```
images/
└── <dataset_name>/
    └── <entity_name>/
        └── <prompt_id>/
            ├── 1.png
            ├── 2.png
            ├── 3.png
            └── ...
```

* `dataset_name`: name of the T2I model
* `entity_name`: entity being analysed
* `prompt_id`: unique ID corresponding to a country-entity pair in country_prompts.csv

## Usage

The pipeline evaluates generated images along four axes using a VQA model (Gemini Flash / Qwen):

    * Entity Appearance (VDI-Entity)
    * Background Appearance (VDI-Background)
    * Affluence (SEVI-Affluence)
    * Maintenance (SEVI-Maintenance)

### Running the Full Pipeline

To run the complete evaluation pipeline:

```bash
bash run.sh <ENTITY> <DATASET> <PROJECT_ID> <VQA_MODEL> <BATCH>
```

**Example:**
```bash
bash run.sh "house" "sd21" "my-project-id" "flash" "1"
```

```markdown
**Parameters:**
- `ENTITY`: The entity to analyze (e.g., "house", "car", "backyard", "dog"). Must be one from those in `country_prompts.csv`.
- `DATASET`: The dataset/model name matching your image folder (e.g., "sd21", "sd3m", "flux1", "sd35")
- `PROJECT_ID`: Google Cloud project ID
- `VQA_MODEL`: (Optional) VQA model to use - "flash" for Gemini Flash (default, supports batch processing) or "qwen" for Qwen VQA model
- `BATCH` : 1 if using batch processing (supported only for flash) or 0 (supported for both qwen and flash)
```

### Pipeline Components

The pipeline executes three main evaluation components:

1. **VDI Entity Appearance** (`scripts/vdi_entity.sh`):
   - Runs visibility checks for image attributes
   - Performs VQA for non-fixed (NF) questions with multiple possible answers
   - Performs VQA for fixed (F) questions with determinate answers
   - Analyzes and computes diversity scores for entity appearance from VQA responses, also saves plots

2. **VDI Background Appearance** (`scripts/vdi_background.sh`):
   - Iteratively analyzes background elements through 4 steps
   - Each step adaptively generates questions based on previous answers
   - Computes diversity scores for background variation, also saves plots

3. **SEVI Dimensions** (`scripts/sevi.sh`):
   - Evaluates affluence axis (impoverished to affluent)
   - Evaluates maintenance axis (worn to pristine)
   - Analyzes and computes scores for both socio-economic dimensions, also saves plots

### Running Individual Components

You can also run individual components separately with the same parameters:

```bash
bash scripts/vdi_entity.sh "house" "sd21" "my-project-id" "flash" "1"   # VDI Entity Appearance only
bash scripts/vdi_background.sh "house" "sd21" "my-project-id" "flash" "1"   # VDI Background Appearance only
bash scripts/sevi.sh "house" "sd21" "my-project-id" "flash" "1"       # SEVI dimensions only
```

### Collating Results

After running the pipeline for all desired entities and datasets, collate all scores into a single file:

```bash
python score_map.py --output_path 'results' --vqa_model VQA_MODEL
```

This aggregates scores from all evaluated entity-country-dataset combinations across all four axes.

## Released Resources

### Question-Answer Set
The question and answer sets released as part of this work are in `resources/question_answer_set.csv`. Details regarding question-answer generation and preprocessing are documented in the `resources/question-answer-generation/` folder, see info.txt.

### Prompts
The prompts used in various stages of VQA can be found in `prompts/` folder and `prompts/info.txt` provides a brief overview.

### Human Annotations
The human annotations obtained as part of our framework validation process can be downloaded from [here](https://drive.google.com/drive/folders/1RGoLVnNeckWPAhtJ1Ft7evYaCtbgBTrN?usp=sharing). This includes the images used in this human validation process.

### Results
The final distributions from our experiments for the 10 entities and 16 countries across 4 T2I models can be downloaded from [here](https://drive.google.com/drive/folders/1QtXcxCzPq8iteq1FehFDjNmLMKZhaLUE?usp=sharing), as well as visualized on the Project Page ([View Examples](https://abhipsabasu.github.io/geodiv/samples.html)).

## Acknowledgements

Parts of this codebase are adapted from [GRADE](https://github.com/RoyiRa/GRADE-Quantifying-sample-diversity-in-text-to-image-models)

## Cite this work

If you find it useful in your research, kindly cite our work:

Basu, A., Singh, M., Agnihotri, S., Keuper, M., and Babu, R. V. *GeoDiv: Framework for Measuring Geographical Diversity in Text-to-Image Models*. In Proc. ICLR, 2026.

```
@inproceedings{basu2026geodiv,
    title={GeoDiv: Framework for Measuring Geographical Diversity in Text-to-Image Models},
    author={Basu, Abhipsa and Singh, Mohana and Agnihotri, Shashank and Keuper, Margret and Babu, R. Venkatesh},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
}
```