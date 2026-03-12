"""
Medical RAG Chatbot
Retrieves relevant medical document chunks from ChromaDB and generates
answers using a local LLM via Ollama.

Usage:
    python chatbot.py
    python chatbot.py --domain cardiac
    python chatbot.py --model mistral:7b
"""

import re
import json
import argparse
import requests
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from config import (
    EMBED_MODEL,
    COLLECTION_NAME,
    CHROMA_PATH,
    DOMAINS,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    RAG_TOP_K,
)
from logger import setup_logger

logger = setup_logger(__name__)


# ─────────────────────────────────────────────
# System prompt — establishes behavior & rules
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a precise medical assistant powered by a retrieval-augmented generation system.

Your role:
- Answer medical questions using ONLY the provided context from the retrieved medical literature.
- Use clear, professional medical language.

RESPONSE FORMAT (STRICT):
- Be CONCISE and DIRECT. Do not use conversational filler ("Based on the provided literature...").
- Synthesize points from multiple sources instead of listing them one by one.
- ALWAYS cite sources inline using this exact format: (Source 1: filename.pdf).
- Do NOT generate a "References", "Bibliography", or "Sources used" section at the end. The system handles this automatically.

Good Example Answer:
Heart failure symptoms include dyspnea on exertion, orthopnea, and lower extremity edema (Source 1: heart-failure.pdf). In severe cases, patients may exhibit elevated jugular venous pressure (Source 2: guidelines.pdf).

Bad Example Answer (DO NOT DO THIS):
Based on the medical literature, heart failure has many symptoms.
Source 1 says it causes dyspnea. Source 2 says it causes edema.
References:
- Source 1: heart-failure.pdf
- Source 2: guidelines.pdf

Important:
- Never provide personal medical advice or diagnosis.
- Put the citation inline immediately after the claim, not at the end of the response."""


# ─────────────────────────────────────────────
# Domain routing keywords
# ─────────────────────────────────────────────
CARDIAC_KEYWORDS = {
    # Conditions & anatomy
    "heart", "cardiac", "cardiovascular", "myocardial", "infarction", "angina",
    "coronary", "arrhythmia", "atrial", "ventricular", "fibrillation",
    "valve", "valvular", "aortic", "mitral", "pericardial", "pericarditis",
    "endocarditis", "cardiomyopathy", "heart failure", "hf", "acs",
    "atherosclerosis", "thrombosis", "embolism", "stenosis",
    # Symptoms (cardiac-leaning)
    "chest pain", "dyspnea", "orthopnea", "edema", "swelling",
    "palpitation", "syncope", "congestion", "breathlessness",
    "leg swelling", "ankle swelling", "jugular",
    # Diagnostics & procedures
    "ecg", "electrocardiogram", "stemi", "nstemi", "troponin", "bnp",
    "ejection fraction", "angioplasty", "stent", "bypass", "cabg",
    "cardiac surgery", "cardiopulmonary", "defibrillator", "pacemaker",
    # Vitals
    "tachycardia", "bradycardia", "hypertension", "hypotension", "murmur",
    "pulmonary edema",
}

GYNAE_KEYWORDS = {
    "pregnancy", "pregnant", "obstetric", "obstetrics", "gynae", "gynecology",
    "gynaecology", "pcos", "polycystic", "endometriosis", "ectopic",
    "pre-eclampsia", "preeclampsia", "eclampsia", "ovarian", "uterine",
    "uterus", "cervical", "cervix", "menstrual", "menstruation", "period",
    "contraception", "contraceptive", "fertility", "infertility", "ivf",
    "miscarriage", "abortion", "placenta", "placental", "fetal", "fetus",
    "trimester", "labor", "labour", "delivery", "cesarean", "c-section",
    "postpartum", "prenatal", "antenatal", "neonatal", "breastfeeding",
    "ovulation", "fallopian", "fibroids", "myoma", "hysterectomy",
    "vulvar", "vaginal", "dysmenorrhea", "amenorrhea", "menarche",
    "menopause", "gestational", "maternal", "obstetric",
}


def detect_domain(question):
    """
    Automatically detect the medical domain of a question using keyword matching.
    Returns 'cardiac', 'gynae', or None (search all domains).
    """
    question_lower = question.lower()

    # Count keyword matches for each domain
    cardiac_score = sum(1 for kw in CARDIAC_KEYWORDS if kw in question_lower)
    gynae_score = sum(1 for kw in GYNAE_KEYWORDS if kw in question_lower)

    logger.debug(f"Domain scores — cardiac: {cardiac_score}, gynae: {gynae_score}")

    # Only route if there's a clear signal (at least 1 match) and one domain wins
    if cardiac_score > 0 and gynae_score == 0:
        return "cardiac"
    elif gynae_score > 0 and cardiac_score == 0:
        return "gynae"
    elif cardiac_score > 0 and gynae_score > 0:
        # Both matched — use the stronger signal
        return "cardiac" if cardiac_score > gynae_score else "gynae" if gynae_score > cardiac_score else None
    else:
        # No keywords matched — search all domains
        return None


def filter_by_majority_domain(results):
    """
    Post-retrieval filter: If one domain dominates the results (>=60%),
    remove chunks from the minority domain to prevent cross-domain noise.
    Only applies when domain was not explicitly set.
    """
    if not results["documents"] or not results["documents"][0]:
        return results

    total = len(results["documents"][0])
    if total <= 1:
        return results

    # Count domain occurrences
    domain_counts = {}
    for meta in results["metadatas"][0]:
        d = meta.get("domain", "unknown")
        domain_counts[d] = domain_counts.get(d, 0) + 1

    # Find the dominant domain
    dominant = max(domain_counts, key=domain_counts.get)
    dominance_ratio = domain_counts[dominant] / total

    # If dominant domain has >= 60% of results, keep only those chunks
    if dominance_ratio >= 0.6 and len(domain_counts) > 1:
        keep_indices = [
            i for i, meta in enumerate(results["metadatas"][0])
            if meta.get("domain") == dominant
        ]

        filtered = {
            "documents": [[results["documents"][0][i] for i in keep_indices]],
            "metadatas": [[results["metadatas"][0][i] for i in keep_indices]],
            "ids": [[results["ids"][0][i] for i in keep_indices]] if results.get("ids") else results.get("ids"),
            "distances": [[results["distances"][0][i] for i in keep_indices]] if results.get("distances") else results.get("distances"),
        }

        removed = total - len(keep_indices)
        logger.info(
            f"Domain filter: kept {len(keep_indices)} {dominant} chunks, "
            f"removed {removed} cross-domain chunk(s)"
        )
        return filtered

    return results


def build_context_block(results):
    """
    Build a formatted context string from ChromaDB retrieval results.
    Each chunk is labeled with its source document and medical domain
    so the LLM can cite them properly.
    """
    context_parts = []

    for i in range(len(results["documents"][0])):
        source = results["metadatas"][0][i].get("source", "Unknown")
        domain = results["metadatas"][0][i].get("domain", "Unknown")
        text = results["documents"][0][i]

        context_parts.append(
            f"[Source {i + 1}: {source} | Domain: {domain}]\n{text}"
        )

    return "\n\n---\n\n".join(context_parts)


def build_user_prompt(context, question):
    """
    Build the final user prompt that combines the retrieved context
    with the user's question in a structured format.
    """
    return f"""Medical Context (retrieved from indexed literature):

{context}

───────────────────────────────────
Patient/Clinician Question:
{question}

Instructions:
1. Provide a direct, concise answer using ONLY the context above.
2. Cite sources inline like this: (Source N: filename.pdf)
3. DO NOT output a references list at the bottom. Start answering immediately without conversational filler."""


def retrieve_chunks(query_text, model, collection, domain_filter=None, top_k=5):
    """
    Convert the user's question into an embedding and retrieve the
    top-k most semantically similar chunks from ChromaDB.

    Uses the BGE instruction prefix for retrieval queries.
    """
    # BGE models use instruction prefixes to distinguish query vs passage embeddings
    query_instruction = "Represent this medical question for retrieval: "

    query_embedding = model.encode(
        query_instruction + query_text, normalize_embeddings=True
    )

    # Apply optional domain filter (e.g., only cardiac or only gynae)
    where_clause = {"domain": domain_filter} if domain_filter else None

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        where=where_clause,
        include=["metadatas", "documents"],
    )

    return results


def generate_answer(context, question, model_name, base_url, temperature, max_tokens):
    """
    Send the context + question to Ollama's local LLM and return the
    generated answer. Uses streaming for real-time output.

    Args:
        context: Formatted context string from retrieved chunks
        question: The user's original question
        model_name: Ollama model identifier (e.g., "llama3.1:8b")
        base_url: Ollama API base URL
        temperature: Sampling temperature (lower = more deterministic)
        max_tokens: Maximum tokens in the response

    Returns:
        The generated answer string, or an error message if generation fails.
    """
    user_prompt = build_user_prompt(context, question)

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        response = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            stream=True,
            timeout=120,
        )
        response.raise_for_status()

        # Stream tokens as they arrive for responsive output
        full_response = []
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    print(token, end="", flush=True)
                    full_response.append(token)

                # Stop if the model signals completion
                if chunk.get("done", False):
                    break

        print()  # Newline after streaming finishes
        return "".join(full_response)

    except requests.exceptions.ConnectionError:
        error_msg = (
            "❌ Cannot connect to Ollama. Make sure it's running:\n"
            "   → Start with: ollama serve\n"
            f"   → Expected at: {base_url}"
        )
        logger.error("Ollama connection failed")
        return error_msg

    except requests.exceptions.Timeout:
        error_msg = "❌ Ollama request timed out. The model may be loading or overloaded."
        logger.error("Ollama request timed out")
        return error_msg

    except requests.exceptions.HTTPError as e:
        error_msg = f"❌ Ollama API error: {e}"
        logger.error(f"Ollama HTTP error: {e}")
        return error_msg

    except Exception as e:
        error_msg = f"❌ Unexpected error during generation: {e}"
        logger.error(f"Generation error: {e}")
        return error_msg


def check_ollama_health(base_url, model_name):
    """
    Verify that Ollama is running and the requested model is available.
    Returns (is_healthy, message).
    """
    try:
        # Check if Ollama server is reachable
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()

        # Check if the requested model is available
        available_models = [m["name"] for m in resp.json().get("models", [])]
        if model_name not in available_models:
            # Also check without tag (e.g., "llama3.1" matches "llama3.1:8b")
            base_names = [m.split(":")[0] for m in available_models]
            if model_name.split(":")[0] not in base_names:
                return False, (
                    f"⚠️  Model '{model_name}' not found in Ollama.\n"
                    f"   Available models: {', '.join(available_models)}\n"
                    f"   Pull it with: ollama pull {model_name}"
                )

        return True, f"✓ Ollama is running with model '{model_name}'"

    except requests.exceptions.ConnectionError:
        return False, (
            "❌ Ollama is not running.\n"
            "   Start it with: ollama serve\n"
            f"   Expected at: {base_url}"
        )
    except Exception as e:
        return False, f"❌ Ollama health check failed: {e}"


def print_sources(results):
    """Print a compact summary of which sources were retrieved."""
    seen = []
    for i in range(len(results["documents"][0])):
        source = results["metadatas"][0][i].get("source", "Unknown")
        domain = results["metadatas"][0][i].get("domain", "Unknown")
        entry = f"  [{domain}] {source}"
        if entry not in seen:
            seen.append(entry)

    print("\n📚 Sources used:")
    for s in seen:
        print(s)


def main():
    # ── Parse CLI arguments ──
    parser = argparse.ArgumentParser(
        description="Medical RAG Chatbot — answers medical questions using retrieved literature"
    )
    parser.add_argument(
        "--domain",
        choices=DOMAINS,
        default=None,
        help="Restrict retrieval to a specific medical domain",
    )
    parser.add_argument(
        "--model",
        default=LLM_MODEL,
        help=f"Ollama model to use for generation (default: {LLM_MODEL})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=RAG_TOP_K,
        help=f"Number of chunks to retrieve (default: {RAG_TOP_K})",
    )
    args = parser.parse_args()

    # ── Step 1: Check Ollama availability ──
    logger.info(f"Checking Ollama health (model: {args.model})...")
    is_healthy, health_msg = check_ollama_health(OLLAMA_BASE_URL, args.model)
    print(health_msg)
    if not is_healthy:
        return

    # ── Step 2: Load the embedding model ──
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL, device=device)
    logger.info("Embedding model loaded")

    # ── Step 3: Connect to ChromaDB ──
    logger.info("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        logger.error(f"Collection '{COLLECTION_NAME}' not found: {e}")
        print(f"\n❌ Collection '{COLLECTION_NAME}' not found.")
        print("Run 'python embed_and_store.py' first to generate embeddings.")
        return

    doc_count = collection.count()
    logger.info(f"Connected to collection: {COLLECTION_NAME} ({doc_count} documents)")

    # ── Print welcome banner ──
    print("\n" + "═" * 60)
    print("  🏥  Medical RAG Chatbot")
    print("═" * 60)
    print(f"  📄 Database : {doc_count} indexed chunks")
    print(f"  🤖 LLM      : {args.model}")
    print(f"  🔍 Top-K    : {args.top_k}")
    if args.domain:
        print(f"  🏷️  Domain   : {args.domain} (forced)")
    else:
        print(f"  🏷️  Domain   : Auto-detect")
    print("═" * 60)
    print("  Type your medical question below.")
    print("  Type 'quit' to exit.\n")

    # ── Step 4: Interactive chat loop ──
    while True:
        try:
            user_question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            logger.info("User exited via interrupt")
            break

        if not user_question:
            continue

        if user_question.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! 👋")
            logger.info("User exited chatbot")
            break

        # ── Step 4a: Determine domain (forced or auto-detected) ──
        if args.domain:
            active_domain = args.domain
        else:
            active_domain = detect_domain(user_question)

        if active_domain:
            logger.info(f"Domain routed to: {active_domain}")
            print(f"  [🏷️ Domain: {active_domain}]")
        else:
            logger.info("No domain detected, searching all domains")
            print(f"  [🏷️ Domain: all]")

        # ── Step 4b: Retrieve relevant chunks ──
        logger.info(f"Retrieving top-{args.top_k} chunks for: {user_question[:80]}...")
        results = retrieve_chunks(
            user_question, embed_model, collection, active_domain, args.top_k
        )

        if not results["documents"] or not results["documents"][0]:
            print("\nBot: I couldn't find any relevant documents for that question.")
            print("     Try rephrasing or removing the domain filter.\n")
            continue

        # ── Step 4c: Apply post-retrieval majority vote filter (if searching all domains) ──
        if not active_domain:
            results = filter_by_majority_domain(results)

        # ── Step 4d: Build context from retrieved chunks ──
        context = build_context_block(results)
        logger.info(
            f"Built context from {len(results['documents'][0])} chunks "
            f"({len(context)} chars)"
        )

        # ── Step 4e: Generate answer via LLM ──
        print("\nBot: ", end="", flush=True)
        answer = generate_answer(
            context,
            user_question,
            model_name=args.model,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

        # ── Step 4f: Show sources ──
        print_sources(results)
        print()

        logger.info(f"Generated answer ({len(answer)} chars)")


if __name__ == "__main__":
    main()
