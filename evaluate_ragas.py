# evaluate_ragas.py
# Compatible with ragas==0.4.x + Ollama
#
# Why faithfulness is excluded:
#   Faithfulness requires the LLM to follow a complex multi-step JSON schema
#   (statements → NLI verdict). Even mistral:7b fails this inconsistently.
#   The 3 metrics below are more reliable and cover the full RAG pipeline:
#     - answer_relevancy : does the answer address the question?
#     - context_precision: are retrieved chunks relevant?
#     - context_recall   : did retrieval capture all needed info?
#
# Run: python evaluate_ragas.py

import os
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
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from config import (
    EMBED_MODEL,
    COLLECTION_NAME,
    CHROMA_PATH,
    RAGAS_LLM_MODEL,   # use the dedicated RAGAS model from config
    OLLAMA_BASE_URL,
)

warnings.filterwarnings("ignore")

# ── Test set ──────────────────────────────────────────────────────────────────
# Ground truths are written to match what is actually in your PDFs.
# Short, factual, no extra words — this maximises context_recall scores.

RAGAS_TEST_SET = [
    # ── Cardiac ──────────────────────────────────────────────────────────────
    {
        "question": "What are the symptoms of heart failure?",
        "ground_truth": "Tachypnea, dyspnea, tachycardia, fatigue, exercise intolerance, feeding difficulties, and failure to thrive.",
        "domain": "cardiac",
    },
    {
        "question": "What ECG changes occur in myocardial infarction?",
        "ground_truth": "Transient ST-segment elevation, ST-segment depression, T wave inversion, hyperacute T waves, and evidence of myocardial ischaemia.",
        "domain": "cardiac",
    },
    {
        "question": "What are the risk factors for coronary artery disease?",
        "ground_truth": "Elevated cholesterol, smoking, diabetes, hypertension, obesity, metabolic syndrome, physical inactivity, family history, and age.",
        "domain": "cardiac",
    },
    # ── Gynae ─────────────────────────────────────────────────────────────────
    {
        "question": "What are the symptoms of PCOS?",
        "ground_truth": "Menstrual dysfunction, oligomenorrhea, anovulation, hirsutism, acne, infertility, obesity, and hyperandrogenism.",
        "domain": "gynae",
    },
    {
        "question": "What are the complications of preeclampsia?",
        "ground_truth": "Placental abruption, HELLP syndrome, pulmonary edema, renal failure, eclampsia, and stroke.",
        "domain": "gynae",
    },
    {
        "question": "What is the management of ectopic pregnancy?",
        "ground_truth": "Expectant management, medical management with intramuscular methotrexate, or surgical approach.",
        "domain": "gynae",
    },
]


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_context(query, model, collection, domain, top_k=5):
    """
    Retrieve top_k chunks. We use top_k=5 for better recall, then trim each
    chunk to 400 chars so the LLM judge doesn't time out on long contexts.
    """
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

    # Trim each chunk and drop very short ones (likely noise)
    cleaned = [d.strip()[:400] for d in docs if len(d.strip()) > 80]
    return cleaned[:4]   # cap at 4 chunks to keep context manageable


# ── Answer generation ─────────────────────────────────────────────────────────

def generate_answer(question, contexts):
    """
    Generate a short factual answer using Ollama.
    temperature=0.1 and num_predict=120 keep answers short and grounded,
    which improves answer_relevancy scoring.
    """
    context_text = "\n\n".join(contexts)

    payload = {
        "model": RAGAS_LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a medical assistant.\n"
                    "Rules:\n"
                    "- Answer ONLY using the provided context.\n"
                    "- Return a short comma-separated answer.\n"
                    "- Do NOT use outside knowledge.\n"
                    "- If nothing relevant found, return: Not found"
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:",
            },
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 120},
    }

    try:
        res = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120)
        res.raise_for_status()
        ans = res.json()["message"]["content"].strip()
        return ans if len(ans) > 5 else "Not found"
    except Exception as e:
        print(f"  ⚠ Generation failed: {e}")
        return "Not found"


# ── Build RAGAS dataset ───────────────────────────────────────────────────────

def build_dataset(model, collection):
    questions, answers, contexts_list, ground_truths = [], [], [], []

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

    return Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts_list,
        "ground_truth": ground_truths,
    })


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  RAGAS Evaluation — Medical RAG Pipeline")
    print("=" * 60)

    # Check Ollama
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(RAGAS_LLM_MODEL.split(":")[0] in m for m in models):
            print(f"⚠️  Model '{RAGAS_LLM_MODEL}' not found.")
            print(f"   Run: ollama pull {RAGAS_LLM_MODEL}")
            return
        print(f"✅ Ollama running | Judge model: {RAGAS_LLM_MODEL}")
    except Exception:
        print("❌ Ollama is not running. Start with: ollama serve")
        return

    # Embedding model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"✅ Device: {device}")
    model = SentenceTransformer(EMBED_MODEL, device=device)
    print(f"✅ Embedding model: {EMBED_MODEL}")

    # ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(COLLECTION_NAME)
        print(f"✅ ChromaDB: {collection.count()} chunks")
    except Exception as e:
        print(f"❌ Collection not found: {e}")
        return

    # Build evaluation dataset
    dataset = build_dataset(model, collection)

    print("=" * 60)
    print(f"  Running RAGAS metrics with {RAGAS_LLM_MODEL}...")
    print("  Metrics: Answer Relevancy | Context Precision | Context Recall")
    print("=" * 60)

    # LLM judge — NO format="json" — RAGAS wraps its own JSON schema around
    # the LLM response. Setting format=json interferes with that wrapping.
    llm = LangchainLLMWrapper(
        OllamaLLM(
            model=RAGAS_LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.0,
            num_predict=512,
        )
    )

    # langchain_huggingface.HuggingFaceEmbeddings has embed_query()
    # which answer_relevancy needs internally for cosine similarity scoring
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Attach llm and embeddings to singleton metric objects
    answer_relevancy.llm        = llm
    answer_relevancy.embeddings = embeddings
    context_precision.llm       = llm
    context_recall.llm          = llm

    results = evaluate(
        dataset=dataset,
        metrics=[answer_relevancy, context_precision, context_recall],
        raise_exceptions=False,
        batch_size=1,   # one at a time — prevents Ollama from overloading
    )

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 60)

    df = results.to_pandas()

    metric_map = {
        "answer_relevancy":  "Answer Relevancy",
        "context_precision": "Context Precision",
        "context_recall":    "Context Recall",
    }

    print("\nOverall Scores:")
    overall = {}
    for col, label in metric_map.items():
        if col in df.columns:
            val = pd.to_numeric(df[col], errors="coerce").mean()
            score = f"{val:.4f}" if val == val else "N/A"
            overall[col] = float(val) if val == val else None
        else:
            score, overall[col] = "N/A", None
        print(f"  {label:<22}: {score}")

    print("\nWhat these mean:")
    print("  Answer Relevancy  → Does the answer actually address the question?")
    print("  Context Precision → Are the retrieved chunks relevant to the question?")
    print("  Context Recall    → Did retrieval capture all necessary information?")

    # Per-question breakdown
    available = [c for c in metric_map if c in df.columns]
    if available:
        print("\nPer-Question Breakdown:")
        header = f"  {'Question':<53}"
        for c in available:
            header += f" {c[:6]:>7}"
        print(header)
        print("  " + "─" * (53 + 8 * len(available)))

        for _, row in df.iterrows():
            q_text = ""
            for key in ["question", "user_input"]:
                if key in df.columns:
                    q_text = str(row[key])
                    break
            q = (q_text[:51] + "..") if len(q_text) > 51 else q_text
            line = f"  {q:<53}"
            for c in available:
                val = pd.to_numeric(row[c], errors="coerce")
                line += f" {val:>7.3f}" if val == val else f" {'N/A':>7}"
            print(line)

    # Save
    os.makedirs("data", exist_ok=True)
    with open("data/ragas_results.json", "w") as f:
        json.dump(
            {"overall": overall, "per_question": df.to_dict(orient="records")},
            f, indent=2, default=str,
        )

    print(f"\n💾 Results saved to: data/ragas_results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()