"""
Assignment 2 PS1 targeted patch.
Append/import this module after the original run_experiments.py definitions.
It overrides generate_self_instruct_dataset and adds Task 4 verification,
Natural Questions dataset loading, and question-specific Task 6 retrieval.

Required existing globals from run_experiments.py:
call_groq_llm, SELF_INSTRUCT_SEEDS, TASK6_QUESTIONS, CURATED_DOMAIN_SNIPPETS,
DATA_DIR, FIGURES_DIR, json, re, time, math, pandas as pd.
"""

import json
import re
import time
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# TASK 5 — robust JSON extraction + genuine LLM Self-Instruct generation
# ---------------------------------------------------------------------------

def _balanced_json_values(text):
    """Yield balanced JSON arrays/objects while respecting quoted strings."""
    for start, opening in enumerate(text):
        if opening not in "[{":
            continue
        closing = "]" if opening == "[" else "}"
        depth, in_string, escaped = 0, False, False
        for pos in range(start, len(text)):
            ch = text[pos]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    yield text[start:pos + 1]
                    break


def _extract_json_array(text):
    """Extract a list of instruction records from common JSON response wrappers."""
    if not text or not isinstance(text, str):
        return []

    candidates = [text.strip()]
    candidates += re.findall(
        r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL
    )
    candidates += list(_balanced_json_values(text))

    for candidate in candidates:
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate.strip())
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue

        if isinstance(parsed, dict):
            for key in ("examples", "samples", "data", "instructions"):
                if isinstance(parsed.get(key), list):
                    parsed = parsed[key]
                    break

        if not isinstance(parsed, list):
            continue

        output = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            record = {
                "instruction": str(item.get("instruction", "")).strip(),
                "input": str(item.get("input", "")).strip(),
                "output": str(item.get("output", "")).strip(),
            }
            if (len(record["instruction"].split()) >= 5
                    and len(record["output"].split()) >= 12):
                output.append(record)
        if output:
            return output
    return []


def generate_self_instruct_dataset(num_target=30):
    """
    Generate exactly num_target model-produced examples from the five seed tasks.
    No procedural fallback is silently added. Raises if quota/API/parser prevents
    the required count, so the run cannot misrepresent fallback data as LLM output.
    """
    print(f"\n[Self-Instruct] Target: {num_target} Groq-generated examples")
    samples, seen = [], set()
    batch_size = 3
    max_batches = (num_target + batch_size - 1) // batch_size
    seed_count = len(SELF_INSTRUCT_SEEDS)

    for batch_idx in range(max_batches):
        remaining = num_target - len(samples)
        if remaining <= 0:
            break

        seed = SELF_INSTRUCT_SEEDS[batch_idx % seed_count]
        requested = min(batch_size, remaining)
        prompt = f"""
You are performing Self-Instruct data generation for a research assistant.
Use this human-authored seed task as the topic and style anchor:
SEED INSTRUCTION: {seed['instruction']}
SEED INPUT: {seed['input']}
SEED OUTPUT: {seed['output']}

Generate exactly {requested} NEW and distinct instruction/input/output examples.
Vary the research operation (explain, compare, derive, critique, diagnose, evaluate).
Do not copy the seed wording. Keep each output technically cautious and identify
assumptions or limitations where relevant. Do not invent citations.

Return ONLY a JSON array with this exact schema:
[
  {{"instruction":"...","input":"...","output":"..."}}
]
All values must be JSON strings. No Markdown fences or commentary.
"""
        response = call_groq_llm(
            prompt,
            system_prompt=(
                "You generate diverse, technically careful synthetic research "
                "instruction data. Return valid JSON only."
            ),
            temperature=0.7,
            max_tokens=1800,
            retries=3,
        )
        raw = response.get("content", "") if isinstance(response, dict) else ""
        parsed = _extract_json_array(raw) if response.get("status") == "success" else []

        accepted = 0
        for item in parsed:
            key = re.sub(r"\W+", " ", item["instruction"].lower()).strip()
            if key in seen:
                continue
            seen.add(key)
            item["seed_id"] = seed["id"]
            item["generation_source"] = "groq_llm"
            samples.append(item)
            accepted += 1
            if len(samples) >= num_target:
                break

        print(
            f"[Self-Instruct][{batch_idx + 1}/{max_batches}] "
            f"status={response.get('status', 'unknown')}; "
            f"parsed={len(parsed)}; accepted={accepted}; "
            f"response_chars={len(raw)}"
        )
        if len(samples) < num_target and response.get("status") != "success":
            time.sleep(min(2 * (batch_idx + 1), 10))

    if len(samples) != num_target:
        counts = Counter(x["generation_source"] for x in samples)
        raise RuntimeError(
            f"Self-Instruct incomplete: {len(samples)}/{num_target} valid "
            f"LLM examples. Provenance={dict(counts)}. Fix quota/parser and rerun "
            "Task 5; no procedural fallback was inserted."
        )

    source_counts = Counter(x["generation_source"] for x in samples)
    assert source_counts == {"groq_llm": num_target}
    print(f"[Self-Instruct] Final: {len(samples)}; provenance={dict(source_counts)}")
    print("[Self-Instruct] Structural/duplicate filters applied; factual quality needs review.")
    return samples


# ---------------------------------------------------------------------------
# DATASET COMPLIANCE — Natural Questions sample (official assignment dataset)
# ---------------------------------------------------------------------------


def load_natural_questions_sample(sample_size=500, split="train"):
    """
    Load only the requested number of Natural Questions records
    using Hugging Face streaming. Avoid downloading the full dataset.
    """
    from datasets import load_dataset
    import pandas as pd
    import json
    from pathlib import Path
    from itertools import islice

    print(f"[DATASET] Loading {sample_size} Natural Questions records using streaming...")

    dataset = load_dataset(
        "google-research-datasets/natural_questions",
        split=split,
        streaming=True
    )

    records = list(islice(iter(dataset), sample_size))

    if not records:
        raise RuntimeError("No Natural Questions records were loaded.")

    nq_df = pd.DataFrame(records)

    output_dir = Path("outputs/datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    nq_df.to_json(
        output_dir / "natural_questions_sample.json",
        orient="records",
        indent=2,
        force_ascii=False
    )

    print(f"[DATASET] Loaded records: {len(nq_df)}")
    print(f"[DATASET] Saved sample to: {output_dir}")

    return nq_df



# ---------------------------------------------------------------------------
# TASK 4 — evidence-supported claim check + constrained correction
# ---------------------------------------------------------------------------

def _normal_tokens(text):
    stop = {
        "the", "a", "an", "and", "or", "of", "to", "in", "is", "are",
        "was", "were", "for", "with", "on", "by", "that", "this", "as",
        "it", "from", "at", "be", "has", "have", "had",
    }
    return {
        t for t in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(t) > 2 and t not in stop
    }


def verify_claim_against_evidence(claim, evidence_items, threshold=0.30):
    """
    Transparent lexical support heuristic, not NLI or factual verification.
    Returns the best overlap and source IDs for audit.
    """
    claim_tokens = _normal_tokens(claim)
    best = {"score": 0.0, "source_id": None, "evidence": ""}
    for item in evidence_items:
        evidence = item.get("claim", "")
        evidence_tokens = _normal_tokens(evidence)
        score = (
            len(claim_tokens & evidence_tokens) / len(claim_tokens)
            if claim_tokens else 0.0
        )
        if score > best["score"]:
            best = {
                "score": score,
                "source_id": item.get("publisher"),
                "evidence": evidence,
            }
    best["supported"] = best["score"] >= threshold
    best["verification_method"] = "lexical_overlap_heuristic_not_entailment"
    return best


def run_task4_claim_verification():
    """
    Run a reproducible verification/correction demonstration on the five
    existing synthetic conflict fixtures. The metric is a lexical proxy.
    """
    results = []
    for scenario in CONFLICT_SCENARIOS:
        evidence = [scenario[f"source_{letter}"] for letter in "ABC"]
        winner = max(evidence, key=lambda x: x["credibility"])

        # Construct test claims: one source-grounded and one deliberately unsupported.
        test_claims = [
            {
                "claim": winner["claim"],
                "expected": "supported",
                "case": "source_grounded",
            },
            {
                "claim": (
                    "The same system has demonstrated unrestricted commercial "
                    "deployment with universally guaranteed performance and zero failures."
                ),
                "expected": "unsupported",
                "case": "deliberately_unsupported_test_claim",
            },
        ]

        before, after = [], []
        for item in test_claims:
            verdict = verify_claim_against_evidence(item["claim"], evidence)
            before.append({
                **item,
                **verdict,
            })
            if verdict["supported"]:
                after.append({
                    **item,
                    **verdict,
                    "correction_action": "retained_with_source_attribution",
                })
            else:
                after.append({
                    **item,
                    **verdict,
                    "correction_action": "removed_from_corrected_answer",
                })

        before_unsupported = sum(not x["supported"] for x in before)
        after_unsupported = sum(
            not x["supported"] for x in after
            if x["correction_action"] != "removed_from_corrected_answer"
        )
        results.append({
            "scenario_id": scenario["id"],
            "topic": scenario["topic"],
            "claims_before": before,
            "claims_after": after,
            "unsupported_claim_count_before": before_unsupported,
            "unsupported_claim_count_after": after_unsupported,
            "unsupported_claim_rate_before_pct": round(
                100 * before_unsupported / len(before), 2
            ),
            "unsupported_claim_rate_after_pct": round(
                100 * after_unsupported / max(1, len(after)), 2
            ),
            "metric_basis": (
                "synthetic_claims_and_lexical_overlap_proxy; "
                "not factual hallucination rate or NLI entailment"
            ),
        })

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "task4_claim_verification.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    import pandas as pd
    pd.DataFrame(results).drop(columns=["claims_before", "claims_after"]).to_csv(
        DATA_DIR / "task4_claim_verification_summary.csv", index=False
    )
    print("[Task 4 verification] Saved task4_claim_verification.json")
    print("[Task 4 verification] Lexical proxy only; do not report as factual hallucination rate.")
    return results


# ---------------------------------------------------------------------------
# TASK 6 — question-specific lexical retrieval for the RAG condition
# ---------------------------------------------------------------------------

def retrieve_task6_evidence(question, top_k=3):
    """Rank the existing curated snippets for each question using token overlap."""
    q_tokens = _normal_tokens(question)
    ranked = []
    for snippet_id, snippet in CURATED_DOMAIN_SNIPPETS.items():
        s_tokens = _normal_tokens(snippet)
        score = len(q_tokens & s_tokens) / max(1, len(q_tokens))
        ranked.append((score, snippet_id, snippet))
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected = [x for x in ranked[:top_k] if x[0] > 0]
    return [
        {"snippet_id": sid, "retrieval_score": round(score, 4), "text": text}
        for score, sid, text in selected
    ]


def run_task6_rag_experiment():
    """
    Additional Task 6 experiment: 5 questions x 3 conditions.
    RAG now retrieves snippets per question; condition 3 remains explicitly
    prompt-specialization simulation, not actual fine-tuning.
    """
    import pandas as pd

    records = []
    for item in TASK6_QUESTIONS:
        question = item["q"]
        qid = item["id"]

        prompt_result = call_groq_llm(
            question,
            system_prompt="You are a general research assistant. State uncertainty.",
            temperature=0.2,
            max_tokens=400,
        )

        retrieved = retrieve_task6_evidence(question, top_k=3)
        evidence_block = "\n\n".join(
            f"[{x['snippet_id']}] {x['text']}" for x in retrieved
        )
        if evidence_block:
            rag_prompt = (
                "Answer using only the retrieved evidence below. "
                "Cite snippet IDs for supported claims. If evidence is insufficient, "
                "say so. Do not infer unsupported facts.\n\n"
                f"QUESTION: {question}\n\nRETRIEVED EVIDENCE:\n{evidence_block}"
            )
        else:
            rag_prompt = (
                f"QUESTION: {question}\n"
                "No relevant passage was retrieved. State that the supplied corpus "
                "does not contain sufficient evidence; do not invent an answer."
            )

        rag_result = call_groq_llm(
            rag_prompt,
            system_prompt="You are an evidence-grounded RAG research assistant.",
            temperature=0.1,
            max_tokens=400,
        )

        specialized_result = call_groq_llm(
            question,
            system_prompt=(
                "You are a domain-specialized research assistant. "
                "Use precise technical language, state assumptions and limitations. "
                "This is prompt specialization; model weights are not updated."
            ),
            temperature=0.2,
            max_tokens=400,
        )

        records.append({
            "question_id": qid,
            "question": question,
            "prompting_status": prompt_result.get("status"),
            "prompting_answer": prompt_result.get("content"),
            "prompting_latency_sec": prompt_result.get("request_latency_sec"),
            "prompting_cost_usd": prompt_result.get("cost"),
            "rag_status": rag_result.get("status"),
            "rag_answer": rag_result.get("content"),
            "rag_latency_sec": rag_result.get("request_latency_sec"),
            "rag_cost_usd": rag_result.get("cost"),
            "retrieved_evidence": retrieved,
            "retrieved_snippet_ids": [x["snippet_id"] for x in retrieved],
            "rag_retrieval_method": "lexical_token_overlap_top3",
            "prompt_specialization_status": specialized_result.get("status"),
            "prompt_specialization_answer": specialized_result.get("content"),
            "prompt_specialization_latency_sec": specialized_result.get("request_latency_sec"),
            "prompt_specialization_cost_usd": specialized_result.get("cost"),
            "fine_tuning_status": "NOT_PERFORMED_PROMPT_SPECIALIZATION_ONLY",
            "quality_score": None,
            "hallucination_rate": None,
            "quality_evaluation_status": "NOT_EVALUATED_REQUIRES_LABELLED_REVIEW",
        })
        print(
            f"[Task 6 RAG] {qid}: retrieved={len(retrieved)} "
            f"snippets { [x['snippet_id'] for x in retrieved] }"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "task6_question_specific_rag.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    pd.DataFrame([
        {k: v for k, v in row.items() if k != "retrieved_evidence"}
        for row in records
    ]).to_csv(DATA_DIR / "task6_question_specific_rag.csv", index=False)
    print("[Task 6 RAG] Saved question-specific retrieval experiment.")
    return records


# ---------------------------------------------------------------------------
# ADD THESE CALLS TO THE MAIN EXECUTION SEQUENCE AFTER ORIGINAL TASKS:
#
# 1) Before run_task5(), call load_natural_questions_sample(sample_size=500)
#    once (or call it in the data-preparation section).
# 2) Existing run_task5() will automatically use the replacement generator
#    if this module is imported after the original function definitions.
# 3) Immediately after run_task4(): run_task4_claim_verification()
# 4) Immediately after run_task6(): run_task6_rag_experiment()
# ---------------------------------------------------------------------------
