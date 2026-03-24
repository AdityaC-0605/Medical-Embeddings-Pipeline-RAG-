# evaluate_ragas.py
# Compatible with ragas==0.4.x + Ollama (no OpenAI required)
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
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL, COLLECTION_NAME, CHROMA_PATH, LLM_MODEL, OLLAMA_BASE_URL
from logger import setup_logger

# suppress deprecation noise — we know ragas.metrics is legacy but it's the
# only path that supports Ollama; ragas.metrics.collections requires OpenAI
warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = setup_logger(__name__)

# ── Test set ──────────────────────────────────────────────────────────────────

RAGAS_TEST_SET = [
    {
        "question": "What are the symptoms of heart failure?",
        "ground_truth": "Shortness of breath, fatigue, lower extremity swelling, orthopnea, and elevated jugular venous pressure.",
        "domain": "cardiac",
    },
    {
        "question": "What ECG changes occur in myocardial infarction?",
        "ground_truth": "ST segment elevation or depression on ECG and evidence of myocardial ischaemia.",
        "domain": "cardiac",
    },
    {
        "question": "What are the risk factors for coronary artery disease?",
        "ground_truth": "Hypertension, diabetes, hypercholesterolaemia, smoking, obesity, and family history.",
        "domain": "cardiac",
    },
    {
        "question": "What are the symptoms of PCOS?",
        "ground_truth": "Irregular periods, hirsutism, acne, obesity, and infertility due to anovulation and hyperandrogenism.",
        "domain": "gynae",
    },
    {
        "question": "What are the complications of preeclampsia?",
        "ground_truth": "Preterm birth, intrauterine growth restriction, maternal end-stage renal disease, chronic hypertension, and neonatal pulmonary dysplasia.",
        "domain": "gynae",
    },
    {
        "question": "What is the management of ectopic pregnancy?",
        "ground_truth": "Surgical treatment via salpingectomy or salpingostomy, or medical management with methotrexate.",
        "domain": "gynae",
    },
]


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_context(query, model, collection, domain, top_k=3):
    embedding = model.encode(
        "Represent this question for searching relevant passages: " + query,
        normalize_embeddings=True,
    )
    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=top_k,
        where={"domain": domain},
        include=["documents"],
    )
    return results["documents"][0] if results["documents"] else []


# ── Answer generation ─────────────────────────────────────────────────────────

def generate_answer(question, contexts):
    context_text = "\n\n---\n\n".join(contexts)
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict medical assistant. "
                    "Answer ONLY using the provided context. "
                    "Return a short factual answer. "
                    "Do NOT explain. Do NOT hallucinate."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context_text}\n\n"
                    f"Question: {question}\n\nAnswer:"
                ),
            },
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 128},
    }
    try:
        res = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120)
        res.raise_for_status()
        return res.json()["message"]["content"].strip()
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return "Error generating answer."


# ── Build RAGAS dataset ───────────────────────────────────────────────────────

def build_dataset(model, collection):
    questions, answers, contexts_list, ground_truths = [], [], [], []

    print(f"\n📦 Building dataset ({len(RAGAS_TEST_SET)} questions)...\n")

    for item in RAGAS_TEST_SET:
        q, gt, d = item["question"], item["ground_truth"], item["domain"]
        print(f"  Q: {q}")

        contexts = retrieve_context(q, model, collection, d)
        if not contexts:
            contexts = ["No context found."]

        answer = generate_answer(q, contexts)
        print(f"  A: {answer[:100]}...\n")

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
        print(f"✅ Ollama running at {OLLAMA_BASE_URL}")
    except Exception:
        print("❌ Ollama is not running. Start with: ollama serve")
        return

    # Embedding model for retrieval
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

    # Build dataset
    dataset = build_dataset(model, collection)

    print("=" * 60)
    print("  Running RAGAS metrics (takes ~5-10 mins)...")
    print("=" * 60)

    # ── LLM judge ────────────────────────────────────────────────────────────
    # format="json" forces Ollama to produce valid JSON output — without this
    # llama3.1:8b produces free-text which fails RAGAS's internal JSON parser
    llm = LangchainLLMWrapper(
        OllamaLLM(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.0,
            num_predict=256,
            format="json",
        )
    )

    # ── Embeddings for answer_relevancy ──────────────────────────────────────
    # FIX: use langchain_huggingface.HuggingFaceEmbeddings — this has
    # the embed_query() method that RAGAS's answer_relevancy metric needs.
    # ragas.embeddings.HuggingFaceEmbeddings does NOT have embed_query().
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Attach to singleton metric objects
    faithfulness.llm            = llm
    answer_relevancy.llm        = llm
    answer_relevancy.embeddings = embeddings
    context_precision.llm       = llm
    context_recall.llm          = llm

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        raise_exceptions=False,
        batch_size=1,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 60)

    df = results.to_pandas()

    metric_map = {
        "faithfulness":      "Faithfulness",
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
            score = "N/A"
            overall[col] = None
        print(f"  {label:<22}: {score}")

    print("\nWhat these mean:")
    print("  Faithfulness      → Is the answer grounded in retrieved context? (no hallucination)")
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

    failed = [label for col, label in metric_map.items() if overall.get(col) is None]
    if failed:
        print(f"\n⚠️  {', '.join(failed)} returned N/A.")
        print("   If format=json didn't help for Faithfulness, try:")
        print("   ollama pull mistral:7b")
        print("   LLM_MODEL=mistral:7b python evaluate_ragas.py")

    print("=" * 60)


if __name__ == "__main__":
    main()