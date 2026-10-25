"""Inspect SQLite persistence tables after Phase 10 API smoke execution.
====================================================================
Prints queries, responses, and retrieval events stored during inference.
"""

from contextforge.storage.db import get_connection
from contextforge.storage.repositories import QueryResultRepository


def inspect_database():
    conn = get_connection()
    repo = QueryResultRepository(conn)

    queries = repo.list_recent_queries(limit=10)
    print("=" * 60)
    print(f"DATABASE INSPECTION: {len(queries)} RECENT QUERIES")
    print("=" * 60)

    for q in queries:
        print(f"\n[Query ID]: {q['query_id']}")
        print(f"  Question:  {q['question'][:80]}...")
        print(f"  Pipeline:  {q['pipeline']}")
        print(f"  Model ID:  {q['model_id']}")
        print(f"  Latency:   {q['latency_ms']:.1f} ms")
        print(f"  Created:   {q['created_at']}")
        print(f"  Response:  {q['response_id']}")

        # If RAG, display retrieval events
        if q["pipeline"] == "rag" and q["response_id"]:
            events = repo.get_retrieval_events(q["response_id"])
            print(f"  Retrieval Events ({len(events)} sources):")
            for e in events:
                print(
                    f"    - Rank {e['rank']}: Chunk ID '{e['chunk_id']}' | "
                    f"Doc '{e['document_id']}' | Score: {e['score']:.4f} | Label: {e['source_label']}"
                )


if __name__ == "__main__":
    inspect_database()
