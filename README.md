# Azerbaijani Sentence Similarity Benchmark

Benchmark testing embedding models in Azerbaijani for sentence similarity tasks. This project evaluates various pre-trained embedding models on multiple Azerbaijani sentence similarity datasets by computing cosine similarities and assessing performance using Pearson correlation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Datasets](#datasets)
- [Models](#models)
- [Parameters](#parameters)
- [Installation](#installation)
- [Usage](#usage)


## Overview

Understanding sentence similarity is crucial for various Natural Language Processing (NLP) applications such as information retrieval, paraphrase detection, and semantic search. This project provides a benchmarking framework to evaluate different embedding models' effectiveness in capturing semantic similarities between Azerbaijani sentences.

## Features

- **Support for Multiple Models:** Evaluate a wide range of pre-trained embedding models.
- **Multiple Datasets:** Test models on various Azerbaijani sentence similarity datasets.
- **Efficient Processing:** Utilize batching and GPU acceleration for faster computations.
- **Comprehensive Metrics:** Calculate Pearson correlation to assess model performance.
- **Easy Integration:** Simple command-line interface for seamless benchmarking.

## Datasets

The benchmarking framework utilizes several Azerbaijani sentence similarity datasets. Each dataset contains pairs of sentences along with human-annotated similarity scores.

### Included Datasets

1. **Azerbaijani STS Benchmark**
   - **Identifier:** `LocalDoc/Azerbaijani-STSBenchmark`
   - **Description:** A standard benchmark for sentence similarity in Azerbaijani, containing diverse sentence pairs.

2. **Azerbaijani BIOSSES STS**
   - **Identifier:** `LocalDoc/Azerbaijani-biosses-sts`
   - **Description:** Based on the BIOSSES dataset, adapted for Azerbaijani language.

3. **Azerbaijani SICK-R STS**
   - **Identifier:** `LocalDoc/Azerbaijani-sickr-sts`
   - **Description:** Adapted from the SICK-R dataset for Azerbaijani sentence similarity tasks.

4. **Azerbaijani STS12 STS**
   - **Identifier:** `LocalDoc/Azerbaijani-sts12-sts`
   - **Description:** Part of the STS12 series, tailored for Azerbaijani.

5. **Azerbaijani STS13 STS**
   - **Identifier:** `LocalDoc/Azerbaijani-sts13-sts`
   - **Description:** Part of the STS13 series, tailored for Azerbaijani.

6. **Azerbaijani STS15 STS**
   - **Identifier:** `LocalDoc/Azerbaijani-sts15-sts`
   - **Description:** Part of the STS15 series, tailored for Azerbaijani.

7. **Azerbaijani STS16 STS**
   - **Identifier:** `LocalDoc/Azerbaijani-sts16-sts`
   - **Description:** Part of the STS16 series, tailored for Azerbaijani.

### Dataset Structure

Each dataset should be structured in a format compatible with the Hugging Face `datasets` library, containing:

- `sentence1`: The first sentence in the pair.
- `sentence2`: The second sentence in the pair.
- `score`: Human-annotated similarity score (use `scaled_score` if applicable).

## Models

To evaluate different embedding models, provide a text file (`models.txt`) listing the Hugging Face model names you wish to benchmark, one per line. Example:


## Parameters

The benchmarking script accepts several parameters to customize the evaluation process:

- `--models_file`: **(Required)** Path to the text file containing the list of model names to evaluate.
- `--output`: **(Optional)** Output CSV file to save results. Defaults to `benchmark_results.csv`.
- `--batch_size`: **(Optional)** Batch size for processing sentences. Defaults to `32`.
- `--device`: **(Optional)** Device to run the model on (`cpu` or `cuda`). Defaults to `cpu`.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/azerbaijani-sentence-similarity-benchmark.git
   cd azerbaijani-sentence-similarity-benchmark
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Dependencies**


## Usage

To evaluate models on Azerbaijani sentence similarity datasets, follow these steps:

### 1. Prepare the Models List

Create a `models.txt` file where each line contains the Hugging Face model name you want to evaluate. Example:

```plaintext
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
sentence-transformers/distiluse-base-multilingual-cased-v1
bert-base-multilingual-cased
xlm-roberta-base
```


### 2. Run the Benchmark Script

   ```bash
   python benchmark.py --models_file models.txt --output results.csv --batch_size 32 --device cuda
   ```


### Command Line Parameters:

- `--models_file`: **(Required)** Path to the `models.txt` file containing a list of model names to evaluate.
- `--output`: **(Optional)** Name of the output CSV file to save results. Defaults to `benchmark_results.csv`.
- `--batch_size`: **(Optional)** Batch size for processing sentences. Larger sizes may improve processing speed but require more memory. Defaults to `32`.
- `--device`: **(Optional)** Device to run the model on (`cpu` or `cuda`). Use `cuda` to enable GPU acceleration if available.


### Example Command

To run the benchmark on GPU with a batch size of 64 and save results to `custom_results.csv`, use the following command:

```bash
python benchmark.py --models_file models.txt --output custom_results.csv --batch_size 64 --device cuda
```

## View the Results

Once the benchmarking completes, the results are saved in the specified output file (default is `results.csv`). This file contains the Pearson correlation scores for each model-dataset pair along with an average score for each model.

#### Example Output Format

| Model                                     | Azerbaijani-STSBenchmark | Azerbaijani-biosses-sts | Azerbaijani-sickr-sts | Average Pearson |
|-------------------------------------------|--------------------------|-------------------------|-----------------------|------------------|
| sentence-transformers/paraphrase-MiniLM   | 0.85                     | 0.80                    | 0.78                  | 0.81             |
| bert-base-multilingual-cased              | 0.78                     | 0.75                    | 0.73                  | 0.75             |
| xlm-roberta-base                          | 0.79                     | 0.77                    | 0.76                  | 0.77             |

- **Individual Scores:** Each dataset column represents the Pearson correlation score between predicted and true similarity scores for the given model.
- **Average Pearson:** This column shows the average Pearson correlation score across all datasets for each model, providing an overall measure of performance.

Use these scores to compare the effectiveness of different models in capturing sentence similarity for Azerbaijani text.
