import json
import argparse
import requests
import re
import string
import numpy as np
import pandas as pd
import time
from typing import List, Dict
from tqdm import tqdm

# ---------------- Utility Functions ---------------- #

def get_model_answer(question: str) -> str:
    url = "http://localhost:8000/ask"
    payload = {"question": question, "session_id": "eval_session"}
    try:
        start_time = time.time()
        response = requests.post(url, json=payload)
        response.raise_for_status()
        elapsed_time = time.time() - start_time
        data = response.json()
        return data.get("answer", "").strip(), elapsed_time
    except Exception as e:
        print(f"Error querying model for question '{question}': {e}")
        return "", 0.0

def normalize_text(text: str) -> str:
    def remove_articles(s): return re.sub(r'\\b(a|an|the)\\b', ' ', s)
    def white_space_fix(s): return ' '.join(s.split())
    def remove_punc(s): return ''.join(ch for ch in s if ch not in set(string.punctuation))
    def lower(s): return s.lower()
    return white_space_fix(remove_articles(remove_punc(lower(text))))

def compute_f1(a_gold: str, a_pred: str) -> float:
    gold_toks = normalize_text(a_gold).split()
    pred_toks = normalize_text(a_pred).split()
    common = set(gold_toks) & set(pred_toks)
    num_same = sum(min(gold_toks.count(tok), pred_toks.count(tok)) for tok in common)
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)

def compute_rouge_l(a: str, b: str) -> float:
    # Simplified ROUGE-L (based on longest common subsequence length)
    a_tokens, b_tokens = a.split(), b.split()
    dp = [[0]*(len(b_tokens)+1) for _ in range(len(a_tokens)+1)]
    for i in range(len(a_tokens)):
        for j in range(len(b_tokens)):
            if a_tokens[i] == b_tokens[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(b_tokens)
    recall = lcs / len(a_tokens)
    return 2 * precision * recall / (precision + recall + 1e-8)

def compute_bleu(a: str, b: str) -> float:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    return sentence_bleu([a.split()], b.split(), smoothing_function=SmoothingFunction().method1)

def compute_bert_score(a: str, b: str) -> float:
    try:
        from bert_score import score
        P, R, F1 = score([b], [a], lang="en", verbose=False)
        return F1[0].item()
    except ImportError:
        return 0.0

def is_similar(pred: str, ref: str, threshold=0.6) -> bool:
    return compute_rouge_l(ref, pred) > threshold

# ---------------- Evaluation Loop ---------------- #

def evaluate(dataset: List[Dict[str, str]]) -> Dict[str, float]:
    metrics = {
        "rouge_l": [],
        "bleu": [],
        "bert_score": [],
        "f1": [],
        "response_time": []
    }

    for entry in tqdm(dataset):
        question = entry['question']
        reference_answer = entry['answer']

        pred_answer, response_time = get_model_answer(question)

        metrics["rouge_l"].append(compute_rouge_l(reference_answer, pred_answer))
        metrics["bleu"].append(compute_bleu(reference_answer, pred_answer))
        metrics["bert_score"].append(compute_bert_score(reference_answer, pred_answer))
        metrics["f1"].append(compute_f1(reference_answer, pred_answer))
        metrics["response_time"].append(response_time)

        print(f"Question: {question}")
        print(f"Reference: {reference_answer}")
        print(f"Predicted: {pred_answer}")
        #print(f"F1: {metrics['f1'][-1]:.2f}, EM: {metrics['exact_match'][-1]}, BLEU: {metrics['bleu'][-1]:.2f}, ROUGE-L: {metrics['rouge_l'][-1]:.2f}, BERTScore: {metrics['bert_score'][-1]:.2f}")
        print(f"F1: {metrics['f1'][-1]:.2f}, BLEU: {metrics['bleu'][-1]:.2f}, ROUGE-L: {metrics['rouge_l'][-1]:.2f}, BERTScore: {metrics['bert_score'][-1]:.2f}")
        print("-----")

    results_df = pd.DataFrame(metrics)
    results_df.to_csv("detailed_evaluation.csv", index=False)

    summary = {k: np.mean(v) for k, v in metrics.items()}
    return summary

# ---------------- CLI Entry ---------------- #

def load_dataset_from_csv(path: str) -> List[Dict[str, str]]:
    df = pd.read_csv(path)
    #df = df[df["Question Type"] == "Descriptive"].sample(5, random_state=42)
    df = df.sample(5, random_state=42)
    df = df.rename(columns={'Question': 'question', 'Answer': 'answer'})
    return df[['question', 'answer']].to_dict(orient='records')

def main():
    parser = argparse.ArgumentParser(description="Evaluate NCERT-Tutor with extended metrics")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to evaluation CSV file")
    args = parser.parse_args()

    dataset = load_dataset_from_csv(args.dataset_path)
    results = evaluate(dataset)

    print("\n=== Evaluation Summary ===")
    for k, v in results.items():
        print(f"{k}: {v * 100:.2f}%" if isinstance(v, float) else f"{k}: {v}")

if __name__ == "__main__":
    main()

