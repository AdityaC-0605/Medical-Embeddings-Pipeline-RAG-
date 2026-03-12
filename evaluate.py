import chromadb
import os
import torch
import json
import math
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL, COLLECTION_NAME, CHROMA_PATH, DOMAINS
from logger import setup_logger

logger = setup_logger(__name__)

EVAL_QUERIES = {
    "cardiac": [
        "What are the symptoms of heart failure?",
        "What are the guidelines for myocardial infarction treatment?",
        "What are the risk factors for cardiovascular disease?",
        "How is heart failure managed?",
        "What are the surgical options for cardiac surgery?",
        "What are the choices for revascularization in diabetes?",
        "What is the segmental approach to diagnosis of congenital heart disease?",
    ],
    "gynae": [
        "What are the symptoms of PCOS?",
        "What are the treatment options for ectopic pregnancy?",
        "What are the causes of pre-eclampsia?",
        "What is endometriosis and how is it treated?",
        "What are the Williams obstetrics guidelines?",
        "What is the practical approach for diagnosis and treatment of von Willebrand disease in pregnancy?",
        "What is Dewhurst's textbook of obstetrics and gynaecology?",
        "What is amniotic fluid embolism?",
        "Who edited the 20th edition of Gynaecology by Ten Teachers?",
        "What is the WHO's programme of work on sexual and reproductive health and rights?",
    ],
}

GROUND_TRUTH = {
    "What are the symptoms of heart failure?": [
        "Heart Disease Full Text.pdf",
        "heart-failure.pdf",
        "Heart Failure Management Guidelines 2022.pdf",
    ],
    "What are the guidelines for myocardial infarction treatment?": [
        "ESC myocardial infarction guidelines.pdf",
        "Myocardial infarction.pdf",
        "guideline aha.pdf",
    ],
    "What are the risk factors for cardiovascular disease?": [
        "cardiovascular.pdf",
        "Heart Disease Full Text.pdf",
    ],
    "How is heart failure managed?": [
        "Heart Failure Management Guidelines 2022.pdf",
        "heart-failure.pdf",
    ],
    "What are the surgical options for cardiac surgery?": [
        "cardiac-surgery.pdf",
        "cardiac_surgery2.pdf",
    ],
    "What are the symptoms of PCOS?": ["polycystic.pdf", "PCOS.pdf", "PCOS2.pdf"],
    "What are the treatment options for ectopic pregnancy?": [
        "Ectopic Pregnancy Overview.pdf"
    ],
    "What are the causes of pre-eclampsia?": ["Pre-eclampsia .pdf"],
    "What is endometriosis and how is it treated?": ["endometri.pdf"],
    "What are the Williams obstetrics guidelines?": ["Williams Obstetrics.pdf"],
    "What are the choices for revascularization in diabetes?": ["Heart Compressed.pdf"],
    "What is the segmental approach to diagnosis of congenital heart disease?": ["heart1.pdf"],
    "What is the practical approach for diagnosis and treatment of von Willebrand disease in pregnancy?": ["AJOG PDF Document.pdf"],
    "What is Dewhurst's textbook of obstetrics and gynaecology?": ["gynaec1.pdf"],
    "What is amniotic fluid embolism?": ["gynaec2.pdf"],
    "Who edited the 20th edition of Gynaecology by Ten Teachers?": ["gynaec3.pdf"],
    "What is the WHO's programme of work on sexual and reproductive health and rights?": ["who.pdf"],
}


def compute_recall_at_k(retrieved_sources, relevant_sources, k=5):
    retrieved = set(retrieved_sources[:k])
    relevant = set(relevant_sources)

    if not relevant:
        return 0.0

    hits = len(retrieved.intersection(relevant))
    return hits / len(relevant)


def compute_precision_at_k(retrieved_sources, relevant_sources, k=5):
    retrieved = set(retrieved_sources[:k])
    relevant = set(relevant_sources)

    if not retrieved:
        return 0.0

    hits = len(retrieved.intersection(relevant))
    return hits / k


def compute_mrr(retrieved_sources, relevant_sources):
    relevant = set(relevant_sources)

    for i, source in enumerate(retrieved_sources, 1):
        if source in relevant:
            return 1.0 / i

    return 0.0


def compute_ndcg(retrieved_docs, relevant_docs, k=5):
    relevant = set(relevant_docs)
    seen_relevant = set()
    dcg = 0.0
    for i, doc in enumerate(retrieved_docs[:k], 1):
        if doc in relevant and doc not in seen_relevant:
            dcg += 1.0 / math.log2(i + 1)
            seen_relevant.add(doc)

    ideal = sum(
        1.0 / math.log2(i + 1)
        for i in range(1, min(len(relevant_docs), k) + 1)
    )

    return dcg / ideal if ideal > 0 else 0.0


def evaluate_retrieval(model, collection, queries, domain=None, k=5):
    results = {"recall_at_k": [], "precision_at_k": [], "mrr": [], "ndcg": []}

    # Build set of queries belonging to target domain using EVAL_QUERIES
    domain_queries = set(EVAL_QUERIES.get(domain, [])) if domain else None

    for query, relevant_sources in queries.items():
        if domain_queries is not None and query not in domain_queries:
            continue

        query_instruction = "Represent this medical question for retrieval: "
        query_embedding = model.encode(
            query_instruction + query, normalize_embeddings=True
        )

        where = {"domain": domain} if domain else None

        retrieved = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=where,
            include=["metadatas"],
        )

        if not retrieved["metadatas"] or not retrieved["metadatas"][0]:
            continue

        retrieved_sources = [m.get("source", "") for m in retrieved["metadatas"][0]]

        results["recall_at_k"].append(
            compute_recall_at_k(retrieved_sources, relevant_sources, k)
        )
        results["precision_at_k"].append(
            compute_precision_at_k(retrieved_sources, relevant_sources, k)
        )
        results["mrr"].append(compute_mrr(retrieved_sources, relevant_sources))
        results["ndcg"].append(compute_ndcg(retrieved_sources, relevant_sources, k))

    return {
        metric: sum(values) / len(values) if values else 0
        for metric, values in results.items()
    }


def main():
    logger.info("Starting evaluation")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL, device=device)

    logger.info("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        logger.error(f"Collection '{COLLECTION_NAME}' not found: {e}")
        print(f"❌ Collection '{COLLECTION_NAME}' not found.")
        return

    print("\n" + "=" * 60)
    print("MEDICAL RAG EVALUATION REPORT")
    print("=" * 60)

    all_metrics = evaluate_retrieval(model, collection, GROUND_TRUTH, k=5)
    print(f"\nOverall Metrics (k=5):")
    print(f"  Recall@5:    {all_metrics['recall_at_k']:.4f}")
    print(f"  Precision@5: {all_metrics['precision_at_k']:.4f}")
    print(f"  MRR:         {all_metrics['mrr']:.4f}")
    print(f"  NDCG@5:      {all_metrics['ndcg']:.4f}")

    print("\n" + "-" * 60)
    print("Per-Domain Metrics:")
    print("-" * 60)

    domain_results = {}
    for domain in DOMAINS:
        domain_metrics = evaluate_retrieval(
            model, collection, GROUND_TRUTH, domain=domain, k=5
        )
        domain_results[domain] = domain_metrics
        print(f"\n{domain.upper()}:")
        print(f"  Recall@5:    {domain_metrics['recall_at_k']:.4f}")
        print(f"  Precision@5: {domain_metrics['precision_at_k']:.4f}")
        print(f"  MRR:         {domain_metrics['mrr']:.4f}")
        print(f"  NDCG@5:      {domain_metrics['ndcg']:.4f}")

    print("\n" + "=" * 60)

    output_file = "data/evaluation_results.json"
    os.makedirs("data", exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(
            {
                "overall": all_metrics,
                "domains": domain_results,
            },
            f,
            indent=2,
        )

    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
