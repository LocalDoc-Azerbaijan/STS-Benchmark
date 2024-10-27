import os
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.stats import pearsonr
from tqdm import tqdm

def compute_sentence_embeddings(sentences, tokenizer, model, batch_size=32, device='cpu'):
    embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Computing embeddings"):
        batch_sentences = sentences[i:i+batch_size]
        inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean pooling
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings.append(batch_embeddings.cpu())
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings

def evaluate_model_on_dataset(model_name, dataset_name, batch_size=32, device='cpu'):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()  # Set model to evaluation mode

    # Load the dataset
    dataset = load_dataset(dataset_name, split='train')  # Assuming 'test' split; adjust if necessary

    # Extract sentences and true scores
    sentences1 = [item['sentence1'] for item in dataset]
    sentences2 = [item['sentence2'] for item in dataset]
    true_scores = [item['score'] for item in dataset]  # Use 'scaled_score' if needed

    # Compute embeddings for all sentences in batches
    print(f"Computing embeddings for model '{model_name}' on dataset '{dataset_name}'...")
    embeddings1 = compute_sentence_embeddings(sentences1, tokenizer, model, batch_size=batch_size, device=device)
    embeddings2 = compute_sentence_embeddings(sentences2, tokenizer, model, batch_size=batch_size, device=device)

    # Compute cosine similarities
    print("Computing cosine similarities...")
    cosine_similarities = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)

    # Convert tensors to lists
    predicted_similarities = cosine_similarities.cpu().numpy().tolist()

    # Compute Pearson correlation
    pearson_corr, _ = pearsonr(predicted_similarities, true_scores)

    return pearson_corr

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark similarity models on Azerbaijani datasets.")
    parser.add_argument('--models_file', required=True, help='Path to the text file containing the list of model names to evaluate.')
    parser.add_argument('--output', default='benchmark_results.csv', help='Output CSV file to save results.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing sentences.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (e.g., "cpu", "cuda").')

    args = parser.parse_args()

    datasets_list = [
        'LocalDoc/Azerbaijani-STSBenchmark',
        'LocalDoc/Azerbaijani-biosses-sts',
        'LocalDoc/Azerbaijani-sickr-sts',
        'LocalDoc/Azerbaijani-sts12-sts',
        'LocalDoc/Azerbaijani-sts13-sts',
        'LocalDoc/Azerbaijani-sts15-sts',
        'LocalDoc/Azerbaijani-sts16-sts'
    ]

    # Read the models list from the specified text file
    if not os.path.isfile(args.models_file):
        print(f"Error: The models file '{args.models_file}' does not exist.")
        exit(1)

    with open(args.models_file, 'r') as f:
        models = [line.strip() for line in f if line.strip()]

    # Check if output file exists; if not, create an empty DataFrame
    if os.path.exists(args.output):
        results_df = pd.read_csv(args.output, index_col=0)
    else:
        results_df = pd.DataFrame()

    for model_name in models:
        print(f"\nEvaluating model: {model_name}")
        model_results = {}

        for dataset_name in datasets_list:
            try:
                pearson_corr = evaluate_model_on_dataset(
                    model_name, dataset_name, batch_size=args.batch_size, device=args.device)
                print(f"Dataset: {dataset_name}, Pearson Correlation: {pearson_corr:.4f}")
                model_results[dataset_name] = pearson_corr
            except Exception as e:
                print(f"Error evaluating on {dataset_name}: {e}")
                model_results[dataset_name] = None

        # Calculate average Pearson correlation
        valid_scores = [v for v in model_results.values() if v is not None]
        average_pearson = sum(valid_scores) / len(valid_scores) if valid_scores else None
        model_results['Average Pearson'] = average_pearson
        model_results['Model'] = model_name

        # Append results to the DataFrame using pd.concat
        results_df = pd.concat([results_df, pd.DataFrame([model_results])], ignore_index=True)

        # Save results to CSV
        results_df.to_csv(args.output, index=False)

    print(f"\nBenchmarking completed. Results saved to {args.output}")

if __name__ == '__main__':
    main()
