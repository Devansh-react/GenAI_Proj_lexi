"""Run progressive, corpus-grounded evaluation cases against the live API.

Usage: python eval/run_eval.py [http://127.0.0.1:8000]
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Any
import json
import sys
import httpx


REFUSAL_STATUS = "insufficient_evidence"


def evaluate_case(client: httpx.Client, base_url: str, case: dict[str, Any]) -> dict[str, Any]:
    """Query one case and return deterministic checks plus the API response."""
    result: dict[str, Any] = {"id": case["id"], "level": case["level"], "question": case["question"]}
    try:
        response = client.post(f"{base_url}/query", json={"question": case["question"]})
        result["http_status"] = response.status_code
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        result.update({"passed": False, "error": str(exc), "checks": {}})
        return result

    expected_answerable = case["expected_answerable"]
    status_ok = payload.get("status") == ("ok" if expected_answerable else REFUSAL_STATUS)
    citations = payload.get("citations", [])
    answer = payload.get("answer", "").lower()
    expected_doc_ok = True
    answer_text_ok = True
    citation_ok = True
    if expected_answerable:
        expected_doc_ok = any(citation.get("doc_id") == case["expected_doc_id"] for citation in citations)
        answer_text_ok = all(term.lower() in answer for term in case["expected_answer_contains"])
        # Every response citation must refer to a chunk returned in this request.
        retrieved_ids = {chunk.get("chunk_id") for chunk in payload.get("retrieval", {}).get("chunks", [])}
        citation_ok = bool(citations) and all(citation.get("chunk_id") in retrieved_ids for citation in citations)
    else:
        citation_ok = not citations
    checks = {"status": status_ok, "expected_document": expected_doc_ok, "answer_content": answer_text_ok, "citations": citation_ok}
    result.update({"passed": all(checks.values()), "checks": checks, "response": payload})
    return result


def run_evaluation(base_url: str = "http://127.0.0.1:8000", cases_path: str | Path = "eval/test_cases.json") -> dict[str, Any]:
    """Run all test cases and return scored metrics for automation or a CLI."""
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))["test_cases"]
    with httpx.Client(timeout=90.0) as client:
        results = [evaluate_case(client, base_url.rstrip("/"), case) for case in cases]
    answerable = [item for item in results if item["id"].startswith("L")]
    refusals = [item for item in results if item["id"].startswith("R")]
    total = len(results)
    passed = sum(item["passed"] for item in results)
    retrieval_hits = sum(item.get("checks", {}).get("expected_document", False) for item in answerable)
    refusal_hits = sum(item.get("checks", {}).get("status", False) for item in refusals)
    citation_hits = sum(item.get("checks", {}).get("citations", False) for item in answerable)
    return {"score_percent": round(100 * passed / total, 1) if total else 0.0, "passed": passed, "total": total, "retrieval_hit_rate": round(retrieval_hits / len(answerable), 3) if answerable else 0.0, "citation_validity": round(citation_hits / len(answerable), 3) if answerable else 0.0, "refusal_accuracy": round(refusal_hits / len(refusals), 3) if refusals else 0.0, "by_level": dict(Counter(f"level_{item['level']}" for item in results if item["passed"])), "results": results}


if __name__ == "__main__":
    report = run_evaluation(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000")
    for item in report["results"]:
        mark = "PASS" if item["passed"] else "FAIL"
        checks = ", ".join(name for name, passed in item.get("checks", {}).items() if not passed)
        print(f"{mark:4} {item['id']:5} level={item['level']} {item['question']}" + (f" | failed: {checks}" if checks else ""))
    print("\nScore: {score_percent}% ({passed}/{total})".format(**report))
    print("Retrieval hit rate: {retrieval_hit_rate:.1%} | Citation validity: {citation_validity:.1%} | Refusal accuracy: {refusal_accuracy:.1%}".format(**report))
