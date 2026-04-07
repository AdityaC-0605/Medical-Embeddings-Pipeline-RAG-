# evaluate_ragas.py
# Optimized for: langchain-core < 0.2.0 and langchain-community
# Run: python evaluate_ragas.py

import os
import re
import json
import warnings
import requests
import chromadb
import torch
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from config import (
    EMBED_MODEL,
    COLLECTION_NAME,
    CHROMA_PATH,
    RAGAS_LLM_MODEL,
    OLLAMA_BASE_URL,
)

warnings.filterwarnings("ignore")

# ── Test set ──────────────────────────────────────────────────────────────────
RAGAS_TEST_SET = [
    {
        "question": "What are the symptoms of heart failure?",
        "ground_truth": "tachypnea, dyspnea, tachycardia, fatigue, exercise intolerance, feeding difficulties, failure to thrive",
        "domain": "cardiac",
    },
    {
        "question": "What ECG changes occur in myocardial infarction?",
        "ground_truth": "ST elevation, ST depression, T wave inversion, hyperacute T waves, myocardial ischaemia on ECG",
        "domain": "cardiac",
    },
    {
        "question": "What are the risk factors for coronary artery disease?",
        "ground_truth": "elevated cholesterol, smoking, diabetes, hypertension, obesity, family history, physical inactivity, metabolic syndrome",
        "domain": "cardiac",
    },
    {
        "question": "How is heart failure managed?",
        "ground_truth": "beta-blockers, ACE inhibitors, diuretics, aldosterone antagonists, angiotensin receptor blockers, medication adjustment",
        "domain": "cardiac",
    },
    {
        "question": "What are the surgical options for cardiac surgery?",
        "ground_truth": "coronary artery bypass grafting, valve replacement, valve repair, cardiac transplantation",
        "domain": "cardiac",
    },
    {
        "question": "What are the symptoms of PCOS?",
        "ground_truth": "oligomenorrhea, anovulation, hirsutism, acne, infertility, obesity, hyperandrogenism, menstrual dysfunction",
        "domain": "gynae",
    },
    {
        "question": "What are the complications of preeclampsia?",
        "ground_truth": "placental abruption, HELLP syndrome, pulmonary edema, renal failure, eclampsia, stroke",
        "domain": "gynae",
    },
    {
        "question": "What is the management of ectopic pregnancy?",
        "ground_truth": "expectant management, methotrexate, salpingectomy, salpingostomy, surgical management",
        "domain": "gynae",
    },
    {
        "question": "What are the causes of preeclampsia?",
        "ground_truth": "placental dysfunction, abnormal trophoblast invasion, hypertension, proteinuria, endothelial dysfunction",
        "domain": "gynae",
    },
    {
        "question": "What are the diagnostic criteria for PCOS?",
        "ground_truth": "oligomenorrhea, hyperandrogenism, polycystic ovaries on ultrasound, two out of three Rotterdam criteria",
        "domain": "gynae",
    },
]

# ── Ollama helper ─────────────────────────────────────────────────────────────
def ollama_call(messages, num_predict=128, temperature=0.0):
    try:
        res = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": RAGAS_LLM_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": num_predict},
            },
            timeout=120,
        )
        res.raise_for_status()
        return res.json()["message"]["content"].strip()
    except Exception as e:
        print(f"  ⚠ Ollama call failed: {e}")
        return ""

# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve_context(query, model, collection, domain, top_k=5):
    embedding = model.encode(
        "Represent this medical question for retrieval: " + query,
        normalize_embeddings=True,
    )
    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=top_k,
        where={"domain": domain},
        include=["documents"],
    )
    docs = results["documents"][0] if results["documents"] else []
    cleaned = [d.strip()[:400] for d in docs if len(d.strip()) > 80]
    return cleaned[:4]

# ── Answer generation ─────────────────────────────────────────────────────────
def generate_answer(question, contexts):
    context_text = "\n\n".join(contexts)
    ans = ollama_call(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical assistant. Answer ONLY using the provided context. "
                    "Return a short comma-separated factual answer. Do NOT use outside knowledge. "
                    "If nothing relevant, return: Not found"
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:",
            },
        ],
        num_predict=120,
        temperature=0.1,
    )
    return ans if len(ans) > 5 else "Not found"

# ── Manual Faithfulness ───────────────────────────────────────────────────────
def compute_faithfulness(answer, contexts):
    if not answer or answer.strip().lower() in ("not found", "insufficient context", ""):
        return None

    context_text = "\n\n".join(contexts)
    parts = re.split(r"[,\n]|\d+[.)]\s*", answer)
    statements = [s.strip().strip(".-•*()") for s in parts if len(s.strip()) > 4]

    if not statements:
        return None

    supported = 0
    for stmt in statements:
        resp = ollama_call(
            messages=[
                {
                    "role": "system",
                    "content": "You are a fact-checker. Reply with only YES or NO.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nStatement: {stmt}\n\nIs this statement supported by context? YES or NO:",
                },
            ],
            num_predict=5,
            temperature=0.0,
        )
        if resp.strip().upper().startswith("YES"):
            supported += 1

    return round(supported / len(statements), 4)

# ── Build RAGAS dataset ───────────────────────────────────────────────────────
def build_dataset(model, collection):
    questions, answers, contexts_list, ground_truths = [], [], [], []
    answer_context_pairs = []

    print(f"\n📦 Building dataset ({len(RAGAS_TEST_SET)} questions)...\n")

    for item in RAGAS_TEST_SET:
        q, gt, d = item["question"], item["ground_truth"], item["domain"]
        print(f"  [{d}] {q}")

        contexts = retrieve_context(q, model, collection, d)
        if not contexts:
            contexts = ["No context found."]

        answer = generate_answer(q, contexts)
        print(f"  → {answer[:100]}\n")

        questions.append(q)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(gt)
        answer_context_pairs.append((answer, contexts))

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    return dataset, answer_context_pairs

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  RAGAS Evaluation — Medical RAG Pipeline")
    print("=" * 60)

    # Ollama Check
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        print(f"✅ Ollama running | Judge model: {RAGAS_LLM_MODEL}")
    except Exception:
        print("❌ Ollama is not running.")
        return

    # ML Setup
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(EMBED_MODEL, device=device)
    print(f"✅ Device: {device} | Embedding: {EMBED_MODEL}")

    # Chroma Check
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(COLLECTION_NAME)
        print(f"✅ ChromaDB: {collection.count()} chunks")
    except Exception as e:
        print(f"❌ Collection not found: {e}")
        return

    # Process Dataset
    dataset, answer_context_pairs = build_dataset(model, collection)

    # ── Step 1: RAGAS Metrics ──
    print("=" * 60)
    print("  Step 1/2 — Running RAGAS metrics...")
    print("=" * 60)

    # Corrected LLM Initialization (No model_kwargs/extra_kwargs)
    llm = LangchainLLMWrapper(
        Ollama(
            model=RAGAS_LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.0
        )
    )
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Assign metrics
    answer_relevancy.llm = llm
    answer_relevancy.embeddings = embeddings
    context_precision.llm = llm
    context_recall.llm = llm

    # Removed batch_size=1 to support Ragas 0.1.9
    results = evaluate(
        dataset=dataset,
        metrics=[answer_relevancy, context_precision, context_recall],
        raise_exceptions=False,
    )

    df = results.to_pandas()

    # ── Step 2: Manual Faithfulness ──
    print("\n" + "=" * 60)
    print("  Step 2/2 — Computing Faithfulness...")
    print("=" * 60)

    faithfulness_scores = []
    for i, (answer, contexts) in enumerate(answer_context_pairs):
        q = RAGAS_TEST_SET[i]["question"]
        print(f"  [{i+1}/{len(answer_context_pairs)}] {q[:55]}...")
        score = compute_faithfulness(answer, contexts)
        faithfulness_scores.append(score)
        status = f"{score:.3f}" if score is not None else "N/A"
        print(f"  Faithfulness: {status}")

    df["faithfulness"] = faithfulness_scores

    # ── Final Results ──
    print("\n" + "=" * 60)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 60)

    metric_map = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
    }

    print("\nOverall Scores:")
    overall = {}
    for col, label in metric_map.items():
        if col in df.columns:
            val = pd.to_numeric(df[col], errors="coerce").mean()
            score = f"{val:.4f}" if val == val else "N/A"
            overall[col] = float(val) if val == val else None
            print(f"  {label:<22}: {score}")

    # Save logic
    os.makedirs("data", exist_ok=True)
    with open("data/ragas_results.json", "w") as f:
        json.dump(
            {"overall": overall, "per_question": df.to_dict(orient="records")},
            f, indent=2, default=str
        )
    print(f"\n💾 Results saved to: data/ragas_results.json")

if __name__ == "__main__":
    main()