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
    OLLAMA_BASE_URL,
)

warnings.filterwarnings("ignore")

LOCAL_EVAL_MODEL = "llama3.1:8b"

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
def ollama_call(messages, num_predict=256, temperature=0.0):
    try:
        res = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": LOCAL_EVAL_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": num_predict},
            },
            timeout=180,
        )
        res.raise_for_status()
        return res.json()["message"]["content"].strip()
    except Exception as e:
        print(f"  ⚠ Ollama call failed: {e}")
        return ""

# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve_context(query, model, collection, domain, top_k=7):
    embedding = model.encode(
        "Represent this sentence for searching relevant passages: " + query,
        normalize_embeddings=True,
    )
    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=top_k,
        where={"domain": domain},
        include=["documents"],
    )
    docs = results["documents"][0] if results["documents"] else []
    cleaned = [d.strip() for d in docs if len(d.strip()) > 80]
    return cleaned

# ── Answer generation ─────────────────────────────────────────────────────────
# FIX: Stronger prompt prevents MCQ-letter answers ("D, G, H") and forces
# full medical sentences so faithfulness decomposition has real claims to check.
def generate_answer(question, contexts):
    context_text = "\n\n".join(contexts)
    ans = ollama_call(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical assistant. Answer ONLY using the provided context. "
                    "Write full sentences describing the medical facts. "
                    "Do NOT return option letters or references. "
                    "Do NOT use outside knowledge. "
                    "If nothing relevant is found in the context, return exactly: Not found"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context_text}\n\n"
                    f"Question: {question}\n\n"
                    "Answer in 2-4 complete medical sentences using only the context above:"
                ),
            },
        ],
        num_predict=200,
        temperature=0.1,
    )
    # Reject answers that look like MCQ option letters (e.g. "A, B, C" or "D, G, H")
    if re.fullmatch(r"[A-J](,\s*[A-J])*\.?", ans.strip()):
        return "Not found"
    return ans if len(ans) > 10 else "Not found"

# ── Proper Faithfulness (NLI-style) ──────────────────────────────────────────
# FIX: The old approach split by commas and checked single words like "dyspnea"
# against huge context chunks — single terms are trivially found, inflating
# faithfulness to 1.0. The correct approach:
#   Step 1 — Ask the LLM to decompose the answer into atomic factual claims
#             (full propositions, not individual words).
#   Step 2 — For each claim, ask: "Can this claim be verified from the context
#             alone, or does it require outside knowledge?"
#   This properly catches hallucinated facts the model adds from its weights.
def decompose_into_claims(answer):
    """Ask LLM to break the answer into atomic factual claims (full sentences)."""
    resp = ollama_call(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise fact extractor. "
                    "Given a medical answer, extract each distinct factual claim as a separate line. "
                    "Each claim must be a complete, self-contained statement — NOT a single word. "
                    "Example bad claim: 'dyspnea' "
                    "Example good claim: 'Dyspnea is a symptom of heart failure.' "
                    "Output ONLY the claims, one per line, no numbering, no extra text."
                ),
            },
            {
                "role": "user",
                "content": f"Medical answer:\n{answer}\n\nExtract atomic factual claims (one per line):",
            },
        ],
        num_predict=300,
        temperature=0.0,
    )
    claims = [
        line.strip().strip("-•*")
        for line in resp.splitlines()
        if len(line.strip()) > 15  # discard single-word lines
    ]
    return claims


def verify_claim_against_context(claim, context_text):
    """
    Ask: can this claim be directly verified from the context?
    Returns True (supported) or False (not supported / hallucinated).
    """
    resp = ollama_call(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict medical fact-checker. "
                    "You will be given a context and a claim. "
                    "Decide if the claim can be DIRECTLY verified using ONLY the provided context. "
                    "Reply with only YES (supported by context) or NO (not supported / requires outside knowledge)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context_text}\n\n"
                    f"Claim: {claim}\n\n"
                    "Is this claim directly supported by the context above? "
                    "Answer YES or NO only:"
                ),
            },
        ],
        num_predict=5,
        temperature=0.0,
    )
    return resp.strip().upper().startswith("YES")


def compute_faithfulness(answer, contexts):
    """
    Proper NLI-style faithfulness:
      faithfulness = (# claims supported by context) / (total # claims)
    Returns None if answer is empty/not-found (excluded from mean).
    """
    if not answer or answer.strip().lower() in ("not found", "insufficient context", ""):
        return None

    context_text = "\n\n".join(contexts)

    # Step 1: decompose into atomic claims
    claims = decompose_into_claims(answer)
    if not claims:
        # Fallback: treat whole answer as one claim
        claims = [answer.strip()]

    # Step 2: verify each claim
    supported = 0
    for claim in claims:
        if verify_claim_against_context(claim, context_text):
            supported += 1

    score = round(supported / len(claims), 4)
    return score

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
        print(f"  → {answer[:120]}\n")

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
    print("  RAGAS Evaluation — Medical RAG Pipeline (Local Ollama)")
    print("=" * 60)

    # Ollama Check
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        print(f"✅ Ollama running | Judge model: {LOCAL_EVAL_MODEL}")
    except Exception:
        print("❌ Ollama is not running. Please start it with 'ollama serve'")
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

    # Build dataset
    dataset, answer_context_pairs = build_dataset(model, collection)

    # ── Step 1: RAGAS Metrics ──
    print("=" * 60)
    print("  Step 1/2 — Running RAGAS metrics (answer_relevancy, context_precision, context_recall)...")
    print("=" * 60)

    llm = LangchainLLMWrapper(
        Ollama(
            model=LOCAL_EVAL_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.0,
        )
    )
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    answer_relevancy.llm = llm
    answer_relevancy.embeddings = embeddings
    context_precision.llm = llm
    context_recall.llm = llm

    results = evaluate(
        dataset=dataset,
        metrics=[answer_relevancy, context_precision, context_recall],
        raise_exceptions=False,
    )

    df = results.to_pandas()

    # ── Step 2: Proper Faithfulness ──
    print("\n" + "=" * 60)
    print("  Step 2/2 — Computing Faithfulness (NLI-style decomposition)...")
    print("  NOTE: A score of 1.0 is EXPECTED for grounded answers. A score < 1.0")
    print("  means the model added claims not supported by the retrieved context.")
    print("=" * 60)

    faithfulness_scores = []
    for i, (answer, contexts) in enumerate(answer_context_pairs):
        q = RAGAS_TEST_SET[i]["question"]
        domain = RAGAS_TEST_SET[i]["domain"]
        print(f"\n  [{i+1}/{len(answer_context_pairs)}] [{domain}] {q[:55]}...")
        print(f"  Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
        score = compute_faithfulness(answer, contexts)
        faithfulness_scores.append(score)
        status = f"{score:.3f}" if score is not None else "N/A (answer was empty/not-found)"
        print(f"  → Faithfulness: {status}")

    df["faithfulness"] = faithfulness_scores
    df["domain"] = [item["domain"] for item in RAGAS_TEST_SET]

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

    print("\n📊 Overall Scores (excluding NaN):")
    overall = {}
    for col, label in metric_map.items():
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            val = numeric.mean()
            n_valid = numeric.notna().sum()
            n_total = len(df)
            score_str = f"{val:.4f}" if pd.notna(val) else "N/A"
            overall[col] = float(val) if pd.notna(val) else None
            print(f"  {label:<22}: {score_str}  (n={n_valid}/{n_total})")

    # Per-domain breakdown
    print("\n📊 Per-Domain Scores:")
    domain_results = {}
    for domain in df["domain"].unique():
        domain_df = df[df["domain"] == domain]
        domain_results[domain] = {}
        print(f"\n  {domain.upper()}:")
        for col, label in metric_map.items():
            if col in domain_df.columns:
                numeric = pd.to_numeric(domain_df[col], errors="coerce")
                val = numeric.mean()
                score_str = f"{val:.4f}" if pd.notna(val) else "N/A"
                domain_results[domain][col] = float(val) if pd.notna(val) else None
                print(f"    {label:<22}: {score_str}")

    # Faithfulness integrity check
    faith_numeric = pd.to_numeric(df["faithfulness"], errors="coerce")
    if faith_numeric.mean() == 1.0:
        print("\n⚠️  WARNING: Faithfulness = 1.0 — this may indicate the LLM judge")
        print("   is not being strict enough. Consider using a larger judge model.")
    elif faith_numeric.mean() >= 0.85:
        print("\n✅ Faithfulness looks reasonable (> 0.85). Your RAG is well-grounded.")
    else:
        print(f"\n⚠️  Faithfulness = {faith_numeric.mean():.3f} — some answers contain")
        print("   claims not supported by the retrieved context (hallucinations).")

    # Save
    os.makedirs("data", exist_ok=True)
    output_path = "data/ragas_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "overall": overall,
                "domains": domain_results,
                "per_question": df.to_dict(orient="records"),
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\n💾 Results saved to: {output_path}")

if __name__ == "__main__":
    main()