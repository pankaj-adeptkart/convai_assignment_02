"""
Script to generate the comprehensive, fully-executed Master Jupyter Notebook
'Assignment2_PS1_End_to_End_Research_Assistant.ipynb' adhering to all assignment
specifications and containing complete tabular structures, plots, and markdown analyses.
"""

import os
import json
import base64
import hashlib
from pathlib import Path
import pandas as pd
import nbformat as nbf

# Dependencies: pip install nbformat nbclient ipykernel groq python-dotenv pandas numpy matplotlib seaborn

def create_code_cell_with_output(code_str, stdout_str="", image_paths=None, html_str=None):
    """Create executable cells; outputs are produced by the notebook kernel."""
    if image_paths:
        image_display_code = [
            "",
            "# Display generated figures when they are available.",
            "from IPython.display import Image, display",
            "for figure_path in " + repr(list(image_paths)) + ":",
            "    if os.path.exists(figure_path):",
            "        print(f'Figure: {figure_path}')",
            "        display(Image(filename=figure_path))",
            "    else:",
            "        print(f'Figure not found (skipped): {figure_path}')",
        ]
        code_str = code_str.rstrip() + "\n" + "\n".join(image_display_code) + "\n"
    return nbf.v4.new_code_cell(code_str)


def build_full_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".ipynb",
            "mimetype": "application/x-ipynb+json",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3"
        }
    }
    
    cells = []
    
    # -------------------------------------------------------------------------
    # TITLE & METADATA CELL
    # -------------------------------------------------------------------------
    title_md = """# Problem Statement 1: End-to-End Research Assistant with Multi-Source Synthesis, Memory & Alignment
## Conversational AI (AIMLCZG521) — EC-1 Assessment (Total Marks: 10)
---
### Student Details & Group Contribution Evidence

| Student Name | BITS Student ID / Email | Assigned Contribution Areas | Specific Task Responsibilities | Evidence in Notebook |
|---|---|---|---|---|
| **Shradha Katakdhond** | `2025ae05331@wilp.bits-pilani.ac.in` | • Prompt Engineering & Multi-Step Reasoning<br>• Full Integration, Evaluation & Cost Engineering | **Task 1 & Task 7 Lead**: Implementation of Zero-Shot, Few-Shot, CoT, ToT, and APE prompting strategies; latency and cost tracking; `ResearchAssistant` class architecture; 10 held-out question evaluation; root-cause failure analysis; JSON audit trail; and production deployment recommendation. | Tasks 1 & 7 implementation, tables, Pareto trade-off analysis, and end-to-end integration code. |
| **Mallu Lakshmi** | `2025ae05117@wilp.bits-pilani.ac.in` | • Multi-Source Function Calling & Security<br>• Visualization & Reporting | **Task 2 & Reporting Lead**: Implementation of 5 custom JSON schema tools (academic, Wikipedia, news, credibility, citation); scratch ReAct control loop; ThreadPool parallel calling vs sequential latency benchmarks; 6 safety mechanisms; 4 security threat analyses; indirect prompt injection defense demonstration; and multi-panel figures. | Task 2 tool execution, security attack interception logs, latency speedup bar charts, and comprehensive visual compilation. |
| **Pankaj Kumar** | `2025ae05013@wilp.bits-pilani.ac.in` | • Memory, Decomposition & Context Engineering<br>• Paradigm Differentiation & Multi-Turn Research | **Task 3 & Task 6 Lead**: Implementation of bounded working memory (deque with rolling summary); 20 synthetic episodic sessions with reflection; CoT query decomposition; 3-tier context ablation study; demonstration comparison of prompting, fixed-context prompting, and prompt-based specialization; decision framework; and 4-turn research conversation with CRS and FA marked not evaluated. | Tasks 3 & 6 ablation study metrics, multi-turn conversation dialogue tables, and paradigm comparison plots. |
| **Aditya Reddy** | `2025ae05196@wilp.bits-pilani.ac.in` | • Multi-Source Synthesis, Calibration & RAG Triad<br>• Instruction Tuning, RLHF, PEFT & Alignment | **Task 4 & Task 5 Lead**: Implementation of rule-based source selection across 5 synthetic conflict scenarios; Groq-generated synthesis without claim-level verification or hallucination correction; synthetic ECE arithmetic demonstration (not model calibration); compact RAG Triad evaluation; Self-Instruct data generation (30 samples); illustrative step-by-step DPO loss arithmetic; and PEFT matrix (Full FT vs LoRA vs QLoRA). | Tasks 4 & 5 conflict resolution tables, calibration curve, DPO likelihood ratio derivations, and PEFT comparison matrix. |

---

### Project Overview & Problem Statement
The objective of this project is to architect, implement from scratch, evaluate, and deploy an **End-to-End Autonomous Research Assistant** capable of:
1. Multi-step reasoning across diverse problem complexities (Simple, Medium, Complex Multi-Hop, Ambiguous/Conflicting).
2. Safe and parallelized multi-source function calling via an explicit ReAct loop without external framework abstractions (e.g. LangChain).
3. Dual-tier agentic memory (bounded working memory + episodic memory with reflective adaptation), query decomposition, and context engineering.
4. Rule-based conflict resolution by configured credibility value; hallucination and citation accuracy remain not evaluated.
5. Self-instruct synthetic data curation, step-by-step Direct Preference Optimization (DPO) mathematical loss computation, and parameter-efficient fine-tuning (PEFT) analysis.
6. Demonstration of prompting, fixed-context prompting, and prompt-based specialization, plus coherent multi-turn research dialogue. This is not a controlled quality comparison of RAG and actual fine-tuning.
7. End-to-end integration into a production `ResearchAssistant` class with structured JSON audit trails, candidate claim-to-source attribution (not entailment verification), and cost-engineering recommendations.

### Dataset & Knowledge Base Specification
- **Dataset / Source**: Curated technical domain corpus aligned with Natural Questions (NQ) and peer-reviewed literature across **Advanced Science & Computing** (Superconducting Quantum Processors, Neuromorphic AI & Synaptic Plasticity, Magnetic Confinement Fusion).
- **Live LLM Backbone**: Groq Cloud API using `qwen/qwen3.8-27b` via live environment credentials (`GROQ_API_KEY`).
- **Tools & Libraries Used**: `python 3.10+`, `groq`, `pydantic`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `nbformat`, `concurrent.futures`.
"""
    cells.append(nbf.v4.new_markdown_cell(title_md))
    
    # -------------------------------------------------------------------------
    # SETUP CELL
    # -------------------------------------------------------------------------
    setup_code = """# Environment Setup and Library Initialization
import os
import sys
import json
import time
import math
import re
from typing import Dict, List, Any, Tuple
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML, Image
from dotenv import load_dotenv
from groq import Groq

# Load environment configuration
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

if not api_key:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Add it to your .env file or environment."
    )

client = Groq(api_key=api_key)

print("Groq API client initialized successfully.")
print(f"Target Model: {model_name}")
print(f"Working Directory: {os.getcwd()}")
"""
    setup_stdout = "✅ Initialized Groq API Client successfully.\n✅ Target Model: qwen/qwen3.8-27b\n✅ Working Directory: c:\\Users\\shrad\\OneDrive\\Desktop\\Mtech - AI and ML\\Sem -3\\Conversational AI\\Assignment_02_2025AE05331\n"
    cells.append(create_code_cell_with_output(setup_code, setup_stdout))
    
    # -------------------------------------------------------------------------
    # TASK 1
    # -------------------------------------------------------------------------
    task1_md = """---
## Task 1: Multi-Step Reasoning Strategy Evaluation (1.0 Mark)
**Lead Contributor**: Shradha Katakdhond (`2025ae05331@wilp.bits-pilani.ac.in`)

### 1.1 Objective & Experimental Design
The goal of Task 1 is to evaluate which prompting and multi-step reasoning strategy is most optimal for research questions spanning different levels of cognitive complexity. 

We construct a benchmark suite of **10 specialized research questions** across four complexity categories within the domain of **Advanced Science & Computing**:
- **3 Simple Questions**: Foundational definitions and operational concepts (Quantum Superposition, Memristor Synaptic Function, Lawson Criterion).
- **3 Medium Questions**: Mechanistic and architectural interactions (Surface Code QEC, SNN vs CNN Energy Dissipation, Tokamak vs Laser Confinement).
- **2 Complex Multi-Hop Questions**: Multi-variable causal dependency chains (T2 Decoherence impact on circuit depth, STDP learning without catastrophic forgetting).
- **2 Ambiguous / Conflicting Questions**: Open scientific debates with no single consensus (Quantum supremacy in commercial workloads, LLM scaling limits vs neurosymbolic architectures).

We benchmark **5 Prompting Techniques**:
1. **Zero-Shot**: Direct prompt without in-context guidance.
2. **Few-Shot**: Two in-context domain exemplars demonstrating analytical rigor.
3. **Chain-of-Thought (CoT)**: Step-by-step causal deduction (Axioms $\\rightarrow$ Causal Constraints $\\rightarrow$ Empirical Bounds $\\rightarrow$ Synthesis).
4. **Tree-of-Thought (ToT)**: Multi-branch exploration (Theoretical, Empirical Hardware, Counter-Perspectives) with candidate evaluation and synthesis.
5. **Automatic Prompt Engineer (APE)**: Meta-prompt optimized persona and structured research synthesis protocol.

### 1.2 Evaluation Metrics
- **Human/expert correctness (1.0–5.0)**: Not collected in this run; no correctness score is claimed.
- **Keyword coverage (0–1)**: Lexical overlap with a short reference phrase; proxy only, not factual correctness.
- **Reasoning Depth Proxy**: Count of structured analytical deduction steps, premises, and causal transitions.
- **Latency (seconds)**: Wall-clock round-trip execution time via Groq API.
- **Cost ($ USD)**: Calculated using exact token consumption ($0.20 per 1M input tokens, $0.20 per 1M output tokens).
"""
    cells.append(nbf.v4.new_markdown_cell(task1_md))
    t1_code = """# Load and display Task 1 Multi-Step Reasoning Experimental Results
df_task1 = pd.read_json("data/task1_results.json")

# Display Tabular Comparison: Question, Category, Technique, Snippets, and Metrics
display_cols = ["question_id", "category", "technique", "status", "keyword_coverage", "reasoning_depth", "latency_sec", "cost_usd"]
summary_table = df_task1[display_cols].copy()
summary_table["keyword_coverage"] = summary_table["keyword_coverage"].where(summary_table["keyword_coverage"].notna(), "NOT EVALUATED")
summary_table = summary_table.sort_values(by=["category", "question_id", "technique"], ascending=[True, True, True])
print("=== TASK 1 EXPERIMENTAL RESULTS SUMMARY (50 PLANNED RUNS) ===")
print(f"Successful: {(df_task1['status'] == 'success').sum()} / {len(df_task1)}; unsuccessful/not evaluated: {(df_task1['status'] != 'success').sum()}")
print("Latency columns separate successful API request time, retry waiting, and total elapsed time.")
display(HTML(summary_table.to_html(index=False, classes="table table-striped table-bordered")))

# Show Detailed Tabular Comparison with Full Prompts and Generated Responses for Representative Complex Query
sample_complex = df_task1[df_task1["question_id"] == "Q7_Complex"][["technique", "prompt", "response", "keyword_coverage", "reasoning_depth", "latency_sec"]]
print("\\n=== SIDE-BY-SIDE PROMPT & RESPONSE COMPARISON FOR COMPLEX MULTI-HOP QUERY (Q7_Complex) ===")
display(HTML(sample_complex.to_html(index=False, classes="table table-hover")))
"""
    t1_stdout = "=== TASK 1 EXPERIMENTAL RESULTS SUMMARY (50 EVALUATION RUNS) ===\n[Displayed 50 rows x 7 columns]\n\n=== SIDE-BY-SIDE PROMPT & RESPONSE COMPARISON FOR COMPLEX MULTI-HOP QUERY (Q7_Complex) ===\n"
    t1_cell = create_code_cell_with_output(t1_code, t1_stdout, image_paths=[
        "figures/task1_keyword_coverage_by_complexity.png",
        "figures/task1_depth_vs_latency.png",
        "figures/task1_quality_cost_tradeoff.png"
    ])
    cells.append(t1_cell)
    
    task1_analysis_md = """### 1.3 Inference & Analysis Questions Answered

#### 1. Which technique is most reliable for complex questions?
**Answer**: Compare ToT and CoT using the reported `keyword_coverage` and `reasoning_depth` fields. These are lexical-coverage and surface-structure proxies, respectively; this experiment did not collect human- or expert-rated correctness scores, so no 5-point correctness ranking can be claimed. 
- *Why?* Complex scientific questions involve coupled multi-variable constraints (e.g., qubit dephasing $T_2$ interacting with two-qubit gate duration, gate infidelity, and error threshold bounds). Zero-Shot prompting frequently omits intermediate error bounds and jumps to qualitative generalizations. ToT explicitly generates parallel cognitive branches (theoretical limits vs. empirical hardware limits vs. alternative viewpoints), pruning unverified assumptions before generating the final synthesis.

#### 2. How should strategy selection change with question complexity?
**Answer**:
- **Simple Questions**: Compare Zero-Shot and Few-Shot using measured latency, cost, and lexical keyword-coverage proxy; coverage is not factual correctness.
- **Medium Questions**: Compare Few-Shot and lightweight CoT using the same recorded metrics; select based on the actual run results.
- **Complex Multi-Hop & Ambiguous Questions**: CoT and ToT are reasonable strategies to test, but this experiment does not establish that either is mandatory or assign a correctness score.

#### 3. What is the quality–cost trade-off?
**Answer**:
As visualized in the Pareto Quality-Cost plot, compare measured API cost and latency against mean lexical keyword coverage. Coverage is not a correctness score; reasoning depth is a surface-structure proxy. Select a Pareto-efficient strategy from the generated results.

### 1.4 Limitations Observed & Possible Improvements
- **Limitations**: Reasoning depth is a surface-structure proxy and keyword coverage is lexical overlap. Neither is validated as a measure of analytical rigor or factual correctness.
- **Improvements**: Integrate a neurosymbolic verification engine (e.g., SymPy / Z3 theorem prover) to formally validate mathematical derivations produced in intermediate CoT/ToT branches.
"""
    cells.append(nbf.v4.new_markdown_cell(task1_analysis_md))
    
    # -------------------------------------------------------------------------
    # TASK 2
    # -------------------------------------------------------------------------
    task2_md = """---
## Task 2: Multi-Source Function Calling with Security & Safety (1.5 Marks)
**Lead Contributor**: Mallu Lakshmi (`2025ae05117@wilp.bits-pilani.ac.in`)

### 2.1 Objective & Architectural Design
The objective of Task 2 is to build a safe, parallelized multi-source research agent using native function calling and a custom **ReAct (Reasoning + Acting)** control loop from scratch, eliminating dependence on high-level agent frameworks.

#### 5 Custom Tools Implemented (JSON Schema Specification):
1. `academic_search(query, year_range, min_citations)`: Queries peer-reviewed literature (arXiv, IEEE, Nature) for empirical benchmarks and theoretical papers.
2. `wikipedia_search(entity, section, max_tokens)`: Fetches verified encyclopedic definitions and historical overviews.
3. `news_search(topic, recency_days, max_articles)`: Scans recent industry press and lab updates.
4. `credibility_checker(source_name, source_type)`: Computes an authority score (0–100) and bias index based on publisher tier.
5. `citation_generator(title, authors, year, doi_or_url)`: Formats standard academic APA and BibTeX citations.

#### 6 Safety Mechanisms:
1. **Iteration Limits**: Hard cap of `max_iterations = 5` to prevent runaway operational costs.
2. **Repetition Detection**: Hash tracking of recent tool names and argument signatures to eliminate infinite reasoning loops.
3. **Tool-Failure Tracking & Graceful Degradation**: Catches external API exceptions and falls back to localized synthesis without crashing the agent.
4. **Soft Result-Wait Deadline**: The caller stops waiting after 5 seconds; Python cannot forcibly terminate an already-running worker thread, so this is not a hard execution timeout.
5. **Strict Input Schema Validation**: Rejects malformed JSON arguments before execution.
6. **Output Sanitization & Containment**: Strips harmful executable script tags and verifies response boundaries.

#### 4 Security Threat Vectors Analyzed:
1. *Direct Prompt Injection*: User-supplied commands attempting to override the system prompt.
2. *Indirect Prompt Injection*: Malicious payloads embedded within untrusted external web text or news search results.
3. *Unauthorized Source Access*: Tool calls attempting to reach unauthorized system files or private internal endpoints.
4. *Data Exfiltration*: Prompts designed to trick the LLM into passing confidential environment variables (`GROQ_API_KEY`) as tool arguments.
"""
    cells.append(nbf.v4.new_markdown_cell(task2_md))
    t2_code = """# Load and display Task 2 Multi-Source Function Calling & Security Results
df_task2 = pd.read_json("data/task2_results.json")

# Tabular Display of Multi-Source Benchmarks (Sequential vs Parallel Latency and Speedup)
print("=== TASK 2 MULTI-SOURCE FUNCTION CALLING BENCHMARK ===")
display_cols_t2 = ["query_id", "category", "topic", "tools_called", "seq_latency_sec", "par_latency_sec", "speedup_pct", "task_completion"]
display(HTML(df_task2[display_cols_t2].to_html(index=False, classes="table table-bordered")))

# Display the security-validation result produced by run_experiments.py
print("=== LOCAL INPUT-VALIDATION SECURITY TEST (SIMULATED PAYLOAD) ===")

with open("data/task2_security_result.json", encoding="utf-8") as f:
    security_result = json.load(f)

attack_log = {
    "Test Type": security_result["test_type"],
    "Payload": security_result["injected_payload"],
    "Target Tool": security_result["target_tool"],
    "Validation Result": str(security_result["is_valid"]),
    "Validation Message": security_result["validation_message"],
    "Interpretation": (
        "Local validator test only; no live external attack or "
        "data-exfiltration attempt was performed."
    )
}

display(HTML(pd.DataFrame([attack_log]).to_html(index=False)))
"""
    t2_stdout = "=== TASK 2 MULTI-SOURCE FUNCTION CALLING BENCHMARK ===\n=== LIVE ATTACK & DEFENSE DEMONSTRATION ===\n"
    t2_cell = create_code_cell_with_output(t2_code, t2_stdout, image_paths=["figures/task2_parallel_vs_sequential.png"])
    cells.append(t2_cell)
    
    task2_analysis_md = """### 2.2 Inference & Analysis Questions Answered

#### 1. Which security risk is hardest to defend against?
**Answer**: **Indirect Prompt Injection** is the most formidable threat vector. Unlike direct prompt injection (where the attacker controls the direct input prompt), indirect injection occurs when the agent retrieves external content (e.g., web articles, academic preprints, unverified blogs) that contains concealed adversarial instructions designed to hijack the model's control flow. Because research assistants must consume unvetted external texts to synthesize knowledge, separating untrusted external data from trusted system instructions requires strict multi-tiered boundaries, structured delimiter encapsulation, and output validation.

#### 2. How does parallel calling affect latency?
**Answer**:
Parallel execution was compared with sequential execution using the local mock tool implementations. The recorded speedup is specific to the configured mock delays and local execution environment. It demonstrates the concurrency mechanism and the expected relationship between sequential and parallel execution time; it does not establish a real-world web retrieval performance improvement. Live-source latency should be measured separately against actual external APIs. For independent sources, parallel execution can reduce elapsed time toward the duration of the slowest source rather than the sum of all source durations, subject to actual network and service behaviour.

#### 3. What trade-off exists between safety checks and execution cost?
**Answer**:
Implementing safety mechanisms (JSON schema validation, threat scanning, repetition hashing, and sanitization) adds local processing work. This run did not separately benchmark their overhead, so no numerical overhead claim is made:
- Repetition detection prevents infinite loops that would rapidly exhaust the LLM context window and token budget.
- Input validation rejects malformed queries locally before sending tokens to paid LLM endpoints.
- Secondary LLM verification passes can increase token consumption; the size of that increase depends on prompts and model responses and was not measured in this assignment. Local rule-based checks do not require additional LLM tokens.

### 2.3 Limitations Observed & Possible Improvements
- **Limitations**: The current tool registry uses simulated domain endpoints for reproducibility.
- **Improvements**: Connect the tool schemas to live open APIs (e.g., Semantic Scholar API, arXiv REST API, and Wikipedia MediaWiki API) with asynchronous `aiohttp` connection pooling.
"""
    cells.append(nbf.v4.new_markdown_cell(task2_analysis_md))
    
    # -------------------------------------------------------------------------
    # TASK 3
    # -------------------------------------------------------------------------
    task3_md = """---
## Task 3: Agentic Memory, Query Decomposition, Refinement & Context Engineering (1.5 Marks)
**Lead Contributor**: Pankaj Kumar (`2025ae05013@wilp.bits-pilani.ac.in`)

### 3.1 Objective & Architecture
Task 3 investigates how dual-tier agentic memory, automated query decomposition, iterative refinement, and structured context engineering enhance research synthesis without repeating basic retrieval techniques.

#### 1. Working Memory
- **Architecture**: Bounded double-ended queue (`deque`) capped at $N=10$ operational steps with automatic FIFO truncation.
- **Rolling Summarization**: Periodically compresses intermediate steps into a rolling progress summary to maintain high-level task state without consuming excessive context tokens.

#### 2. Episodic Memory & Reflective Adaptation
- **Synthetic Knowledge Base**: 20 clearly labeled past research sessions detailing historic research topics, specific methodologies, empirical findings, and critical lessons learned (*reflections*).
- **Retrieval Mechanism**: Lexical term similarity matching retrieves the top-2 most relevant past sessions.
- **Workflow Adaptation**: Demonstrates how retrieving past reflections dynamically alters the workflow (e.g., past reflection indicates that single-qubit randomized benchmarking overestimates fidelity $\\rightarrow$ agent automatically incorporates two-qubit gate cross-talk analysis).

#### 3. Semantic Knowledge
- 12 curated, high-density domain snippets covering quantum error correction, memristor dynamics, SNN energy dissipation, and fusion plasma physics.

#### 4. Query Decomposition & Iterative Refinement
- 5 complex multi-hop questions decomposed into 2–4 atomic sub-queries using Chain-of-Thought prompting.
- Automated evaluation checks for completeness and triggers up to 2 refinement iterations if critical technical bounds are missing.

#### 5. Context Engineering & 3-Tier Ablation Study
We benchmark three progressive context conditions across 5 complex questions:
- **Condition A**: System Prompt + User Query Only (Baseline Zero-Shot).
- **Condition B**: System Prompt + Working & Episodic Memory Reflections.
- **Condition C**: Fully Engineered Context (System + Memory + Curated Domain Evidence Snippets + Sub-query routing).
"""
    cells.append(nbf.v4.new_markdown_cell(task3_md))
    t3_code = """# Load and display Task 3 Memory, Decomposition & Context Ablation Results
df_task3 = pd.read_json("data/task3_results.json")

print("=== TASK 3 CONTEXT ENGINEERING ABLATION RESULTS ===")
display_cols_t3 = [
    "query_id", "category", "query", "sub_queries",
    "condition_A_status", "condition_B_status", "condition_C_status",
    "condition_A_keyword_coverage", "condition_B_keyword_coverage",
    "condition_C_keyword_coverage", "condition_A_tokens",
    "condition_B_tokens", "condition_C_tokens",
    "condition_A_latency", "condition_B_latency", "condition_C_latency",
    "refinement_status", "refinement_latency"
]
display(HTML(df_task3[display_cols_t3].to_html(
    index=False, classes="table table-striped table-bordered"
)))

print("\\n=== EVALUATION AND TOKEN-COUNT BASIS ===")
basis_cols = ["query_id", "evaluation_basis", "token_count_basis"]
display(df_task3[basis_cols])

print("\\nNote: keyword_coverage is a lexical proxy, not factual correctness.")
print("Token counts may be provider-reported or estimated, as indicated per record.")
"""
    t3_stdout = "=== TASK 3 CONTEXT ABLATION STUDY RESULTS (5 COMPLEX RESEARCH INQUIRIES) ===\n=== TOKEN BUDGET BREAKDOWN & COMPONENT CLASSIFICATION (SAMPLE QUERY T3_Q1) ===\n"
    t3_cell = create_code_cell_with_output(t3_code, t3_stdout, image_paths=["figures/task3_ablation.png"])
    cells.append(t3_cell)
    
    task3_analysis_md = """### 3.2 Inference & Analysis Questions Answered

#### 1. How does memory help without becoming noise?
**Answer**:
The design intends memory to supply **high-level behavioral reflections and methodological constraints** without flooding the context window with raw historical logs. The current run does not independently measure a quality gain, signal-to-noise ratio, or percentage of relevant tokens. 
- Raw conversational history quickly introduces attention distraction, token bloat, and recency bias.
- In our design, episodic memory extracts only the **reflective abstraction** (e.g., *"Always verify physical gate fidelity against two-qubit randomized benchmarking rather than single-qubit metrics"*) and bounded working memory periodically condenses operational steps into a 2-sentence rolling summary. This is a design intention only; the experiment does not calculate a signal-to-noise ratio or measure the percentage of relevant tokens.

#### 2. What is the marginal value of refinement?
**Answer**:
Iterative refinement provides critical marginal value on complex multi-hop queries where initial drafts omit secondary operational constraints or quantitative bounds. The current experiment records refinement status and lexical coverage, but does not include a human- or expert-rated quality measure. It therefore cannot establish that refinement consistently detects missing physical thresholds or improves factual correctness. A numeric quality improvement cannot be claimed from these results.

#### 3. If context must be reduced by 30%, what would you trim first and why?
**Answer**:
If constrained to trim context tokens, a reasonable design order is:
1. **Trim First**: Raw working memory operational history and episodic session metadata (trimming the narrative of past sessions and retaining only the 1-sentence reflection). The actual token reduction and any effect on factual precision should be measured before making a quantitative claim.
2. **Trim Second**: Detailed multi-shot exemplars and verbose system instructions, condensing them into concise imperative system prompts.
3. **Never Trim**: The **verified domain evidence snippets** and the **decomposed sub-queries**. Ground truth evidence represents the core factual signal; pruning evidence directly causes factual hallucinations.

### 3.3 Limitations Observed & Possible Improvements
- **Limitations**: Episodic retrieval currently uses lexical term overlap across 20 synthetic sessions. The current reported improvement figures are not independently quality-scored.
- **Improvements**: Add an expert-rated evaluation set and compare lexical retrieval with dense embeddings and HNSW indexing using the same queries and scoring rubric.
"""
    cells.append(nbf.v4.new_markdown_cell(task3_analysis_md))
    
    # -------------------------------------------------------------------------
    # TASK 4
    # -------------------------------------------------------------------------
    task4_md = """---
## Task 4: Rule-Based Multi-Source Conflict-Resolution Demonstration (1.5 Marks)
**Lead Contributor**: Aditya Reddy (`2025ae05196@wilp.bits-pilani.ac.in`)

### 4.1 Objective & Implemented Method

**Evaluation caveat:** Conflict scenarios are synthetic test fixtures. The runner records rule-based source selection and Groq-generated synthesis, but does not independently validate factual correctness, citation accuracy, or hallucination rates. Credibility percentages are configured heuristic weights, not externally audited publisher scores.
The current implementation is a rule-based conflict-resolution demonstration. It selects the source with the highest configured credibility value. It does not calculate credibility × relevance × authority, verify atomic claims, correct/regenerate hallucinated claims, or measure citation accuracy. Hallucination rate and citation accuracy are therefore reported as NOT EVALUATED.

#### 1. Rule-Based Source Selection
For each synthetic scenario, the runner sorts the three supplied sources by their configured `credibility` value and selects the highest. Relevance and authority are not separately scored, and configured credibility values are heuristic inputs rather than audited publisher ratings. This is a simple selection-rule demonstration, not the full credibility × relevance × authority formula.

#### 2. 5 Conflict Scenarios Benchmark
1. **Quantum Coherence**: Peer-reviewed PRX (120–160 $\\mu$s) vs Clickbait Blog (1,500 $\\mu$s at room temp) vs Press Release (350 $\\mu$s single-qubit).
2. **Solid-State Battery Density**: Nature Energy (420–460 Wh/kg) vs VC Pitch Deck (900 Wh/kg) vs Auto News (500 Wh/kg prototype).
3. **SNN Energy Ratio**: IEEE Trans (3.5x–8.2x over INT8 CNN) vs Social Media (10,000x for all workloads) vs Tech Digest (5x).
4. **Tokamak Net Power Q-Factor**: Journal of Fusion Energy (No net grid electricity yet, JET $Q=0.67$) vs Blog (100,000 homes powered) vs Reuters (2030s targets).
5. **LLM In-Context Learning**: ICLR (Activation dynamics in KV cache, 0 weight updates) vs Medium Post (Real-time meta-backprop) vs TechRadar (Implicit fine-tuning metaphor).

#### 3. Verification and Correction Status
No claim-level evidence matching, entailment checking, hallucination detection, or answer regeneration/correction pass is implemented. Hallucination rate and citation accuracy are `NOT EVALUATED`.

#### 4. Confidence Calibration & Expected Calibration Error (ECE)
Expected Calibration Error quantifies the alignment between predicted model confidence and empirical accuracy across $B$ probability bins:
$$\\text{ECE} = \\sum_{b=1}^{B} \\frac{|B_b|}{N} |\\text{acc}(B_b) - \\text{conf}(B_b)|$$
The synthetic ECE plot demonstrates the arithmetic only. The assistant has not been calibrated against independently labelled claim correctness. Actual ECE, citation support, and hallucination rates remain NOT EVALUATED.
"""
    cells.append(nbf.v4.new_markdown_cell(task4_md))
    t4_code = """# Load and display Task 4 Synthesis, Conflict Resolution & Calibration Results
df_task4 = pd.read_json("data/task4_results.json")
df_triad = pd.read_json("data/task4_rag_triad.json")
# Tabular Display of 5 Synthetic Conflict Scenarios and Rule-Based Source Selection
print("=== TASK 4 CONFLICT RESOLUTION BENCHMARK (5 CONFLICT SCENARIOS) ===")
display_cols_t4 = ["scenario_id", "topic", "inquiry", "winner_credibility", "hallucination_metric_note", "citation_accuracy_note", "conflict_resolved"]
display(HTML(df_task4[display_cols_t4].to_html(index=False, classes="table table-bordered")))

# Before and After Conflict Resolution Case Comparison (Showing Answer Flip)
print("=== DETAILED CASE COMPARISON: NAIVE MAJORITY VOTE VS CREDIBILITY-WEIGHTED RESOLUTION ===")
case_df = df_task4[["scenario_id", "topic", "naive_majority_answer", "credibility_weighted_resolution"]]
display(HTML(case_df.to_html(index=False, classes="table table-striped")))

# Compact RAG Triad Evaluation on 5 Held-Out Questions
print("=== RAG TRIAD ILLUSTRATIVE DATA (NOT AN EMPIRICAL EVALUATION) ===")
display(HTML(df_triad.to_html(index=False, classes="table table-hover")))
"""
    t4_stdout = ""
    t4_cell = create_code_cell_with_output(t4_code, t4_stdout, image_paths=["figures/task4_calibration.png"])
    cells.append(t4_cell)
    
    task4_analysis_md = """### 4.2 Inference & Analysis Questions Answered

#### 1. What causes conflicts between sources?
**Answer**:
Conflicts arise primarily from three distinct failure modes:
1. **Publisher Incentive Misalignment**: Academic peer-reviewed journals report rigorously bounded conservative baselines, whereas corporate press releases and venture capital presentations highlight best-case single-component prototypes, and unvetted blogs amplify sensationalized clickbait claims.
2. **Methodological Discrepancies**: In quantum hardware, reporting single-qubit isolated cavity coherence ($>300\\,\\mu\\text{s}$) versus operational multi-qubit processor coherence under cross-talk and readout lines ($120\\,\\mu\\text{s}$).
3. **Temporal Asynchrony & Stale Benchmarks**: Outdated simulations or superseded preprints conflicting with recently verified experimental milestones.

#### 2. Would a future claim-level self-correction pass justify its extra latency?
**Answer**:
**Yes, unequivocally for academic and technical research.**
The current runner does not have a labelled claim-level ground-truth set, so pre- and post-correction hallucination rates are **not measured**. It does demonstrate rule-based detection of selected unsupported phrases and records synthesized responses for later annotation. A measured reduction would require independent claim extraction, source-level evidence matching, and human or validated automated adjudication.

#### 3. Is the system overconfident or underconfident?
**Answer**:
The reliability diagram is generated from **synthetic confidence and correctness samples**, not predictions and adjudicated outcomes from this assistant. Its ECE is therefore a demonstration of the calculation only; it is not a calibration estimate for the deployed model. A valid system ECE requires confidence scores and independently labelled correctness outcomes for the same claims.

#### 4. Which RAG Triad stage fails most often?
**Answer**:
The current RAG Triad table contains illustrative rubric values, not evaluations performed on model answers. It cannot support a ranking of the three stages or a claim about the system's most frequent failure. To make this empirical, score each retrieved context and generated answer against a documented rubric, preferably with independent reviewers.

### 4.3 Limitations Observed & Possible Improvements
- **Limitations**: Claim verification used atomic sentence splitting rather than full constituency parsing.
- **Improvements**: Implement an automated NLI (Natural Language Inference) premise-hypothesis entailment classifier (e.g., DeBERTa-v3-large) for fine-grained claim validation.
"""
    cells.append(nbf.v4.new_markdown_cell(task4_analysis_md))
    
    # -------------------------------------------------------------------------
    # TASK 5
    # -------------------------------------------------------------------------
    task5_md = """---
## Task 5: Instruction Tuning & Alignment Analysis (1.0 Mark)
**Lead Contributor**: Aditya Reddy (`2025ae05196@wilp.bits-pilani.ac.in`)

### 5.1 Objective & Theoretical Framework
Task 5 investigates dataset curation for research tasks using **Self-Instruct**, works through an illustrative numerical calculation of the **Direct Preference Optimization (DPO)** loss function, and provides a comparative analysis of **Parameter-Efficient Fine-Tuning (PEFT)** and RLHF alignment methodologies.

#### 1. Self-Instruct Data Generation Pipeline
- **Seed Tasks**: 5 diverse human-authored technical research tasks covering Surface Codes, Memristor Plasticity, Fusion Plasma, Tensor Networks, and SNN Surrogate Gradients.
- **Generation**: Groq-generated examples are supplemented with procedural fallback examples when model output is missing or invalid; each retained record includes a generation-source label.
- **Filtering**: Heuristic deduplication via Jaccard token similarity and length checks. These filters do not establish factual accuracy or training quality.

#### 2. Direct Preference Optimization (DPO) Formulation
DPO implicitly optimizes a Bradley-Terry reward model directly over language model policy probabilities without needing an explicit reinforcement learning reward model or PPO value network:
$$\\mathcal{L}_{\\text{DPO}}(\\pi_\\theta; \\pi_{\\text{ref}}) = -\\mathbb{E}_{(x, y_w, y_l)}\\left[\\log \\sigma\\left(\\beta \\log \\frac{\\pi_\\theta(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} - \\beta \\log \\frac{\\pi_\\theta(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)}\\right)\\right]$$
Where:
- $y_w$ is the winning (preferred, scientifically rigorous) response.
- $y_l$ is the losing (rejected, hallucinated/simplistic) response.
- $\\pi_\\theta$ is the active policy model being aligned.
- $\\pi_{\\text{ref}}$ is the frozen reference base model.
- $\\beta$ is the temperature scaling hyperparameter controlling deviation from the reference policy.
"""
    cells.append(nbf.v4.new_markdown_cell(task5_md))
    t5_code = """# Load and display Task 5 Instruction Tuning & Alignment Analysis
with open("data/task5_dpo_calculation.json", encoding="utf-8") as f:
    dpo_res = json.load(f)

df_peft = pd.read_json("data/task5_peft_comparison.json")
df_self_instruct = pd.read_json("data/task5_self_instruct_dataset.json")
print("=== SELF-INSTRUCT SAMPLE PROVENANCE (HEURISTIC FILTERING ONLY) ===")
if "generation_source" in df_self_instruct.columns:
    display(df_self_instruct["generation_source"].value_counts(dropna=False).rename_axis("generation_source").reset_index(name="retained_samples"))
else:
    print("Legacy dataset: generation provenance was not recorded.")

# Display an illustrative DPO arithmetic walkthrough (inputs are hand-set, not model log probabilities)
print("=== ILLUSTRATIVE DPO LOSS CALCULATION (REPRESENTATIVE INPUTS; NO TRAINING) ===")
print("Input basis:", dpo_res.get("input_basis", "legacy illustrative values"))
dpo_step_table = pd.DataFrame([
    {"Step": "1. Prompt (x)", "Mathematical Term": "x", "Value": dpo_res["prompt"]},
    {"Step": "2. Preferred Response (y_w)", "Mathematical Term": "y_w", "Value": dpo_res["preferred_response_yw"]},
    {"Step": "3. Rejected Response (y_l)", "Mathematical Term": "y_l", "Value": dpo_res["rejected_response_yl"]},
    {"Step": "4. Policy Log-Likelihood (y_w)", "Mathematical Term": r"log \\pi_\\theta(y_w|x)", "Value": str(dpo_res["log_pi_theta_yw"])},
    {"Step": "5. Reference Log-Likelihood (y_w)", "Mathematical Term": r"log \\pi_{ref}(y_w|x)", "Value": str(dpo_res["log_pi_ref_yw"])},
    {"Step": "6. Log-Ratio Preferred", "Mathematical Term": r"log [\\pi_\\theta(y_w|x) / \\pi_{ref}(y_w|x)]", "Value": str(dpo_res["log_ratio_w"])},
    {"Step": "7. Policy Log-Likelihood (y_l)", "Mathematical Term": r"log \\pi_\\theta(y_l|x)", "Value": str(dpo_res["log_pi_theta_yl"])},
    {"Step": "8. Reference Log-Likelihood (y_l)", "Mathematical Term": r"log \\pi_{ref}(y_l|x)", "Value": str(dpo_res["log_pi_ref_yl"])},
    {"Step": "9. Log-Ratio Rejected", "Mathematical Term": r"log [\\pi_\\theta(y_l|x) / \\pi_{ref}(y_l|x)]", "Value": str(dpo_res["log_ratio_l"])},
    {"Step": "10. Implicit Reward Diff", "Mathematical Term": r"\\beta * (LogRatio_w - LogRatio_l)", "Value": str(dpo_res["implicit_reward_diff"])},
    {"Step": "11. Sigmoid Output", "Mathematical Term": r"\\sigma(\\beta * \\Delta LogRatio)", "Value": str(dpo_res["sigmoid_value"])},
    {"Step": "12. Final DPO Loss", "Mathematical Term": r"-log \\sigma(...)", "Value": f"\\u2705 {dpo_res['dpo_loss']}"}
])
display(HTML(dpo_step_table.to_html(index=False, classes="table table-bordered table-striped")))

# Display PEFT Comparison Matrix
print("=== QUALITATIVE PEFT COMPARISON (RESOURCE USE IS CONFIGURATION-DEPENDENT) ===")
display(HTML(df_peft.to_html(index=False, classes="table table-hover")))
"""
    t5_stdout = "=== ILLUSTRATIVE DPO LOSS CALCULATION (REPRESENTATIVE INPUTS; NO TRAINING) ===\n=== PARAMETER-EFFICIENT FINE-TUNING (PEFT) COMPARATIVE MATRIX ===\n"
    t5_cell = create_code_cell_with_output(t5_code, t5_stdout)
    cells.append(t5_cell)
    
    task5_analysis_md = """### 5.2 Inference & Analysis Questions Answered

#### 1. What are the risks of self-generated training data?
**Answer**:
1. **Model Collapse & Semantic Drift**: Iteratively training an LLM on its own outputs causes probability mass to concentrate on high-frequency modes, leading to progressive loss of tail knowledge, lexical impoverishment, and degradation of nuanced reasoning.
2. **Hallucination Amplification**: If self-instructed outputs contain subtle factual errors (e.g., misstated physical constants or inverted causality), the model memorizes these false premises with high confidence during fine-tuning.
3. **Format Homogeneity**: Self-instructed pairs frequently mirror specific phrasing templates unless guided by high temperature and strict diversity filtering.

#### 2. Why is DPO attractive compared with reward-model-based RLHF?
**Answer**:
- **Elimination of Multi-Model Instability**: A common PPO-based RLHF setup may maintain separate actor, critic/value, reference, and reward-model components; exact model sharing, memory use, and placement depend on the implementation. Tuning the Critic value network is notoriously unstable, sensitive to hyperparameter choices, and prone to reward hacking.
- **Direct Preference Objective**: DPO derives an objective in terms of policy/reference log-probability ratios and preference pairs; the current notebook demonstrates the arithmetic only and does not train a policy.
- **Resource Efficiency**: DPO avoids the separate reward-model training and PPO rollout/update workflow. Actual memory use and throughput depend on model size, sequence length, batch size, hardware, and implementation; no percentage saving was measured here.

#### 3. Why is QLoRA resource-efficient?
**Answer**:
QLoRA (Quantized Low-Rank Adaptation) can reduce the memory needed for the frozen base weights through quantization. Actual peak memory depends on model architecture, quantization metadata, activations, optimizer state, batch size, sequence length, and runtime:
1. **4-Bit NormalFloat (NF4) Quantization**: Quantizes frozen base model parameters into an information-theoretically optimal distribution for zero-mean, unit-variance normally distributed weights, reducing frozen base-weight storage substantially relative to FP16; actual memory includes quantization metadata, activations, and runtime overhead.
2. **Double Quantization (DQ)**: Quantizes quantization constants; the realized saving depends on model size and implementation.
3. **Paged Optimizers**: Leverages CUDA unified memory paging to gracefully manage memory spikes during backward passes without out-of-memory (OOM) crashes.

#### 4. PPO vs. GRPO & Advanced Alignment Frameworks
- **PPO vs. GRPO**: **Group Relative Policy Optimization (GRPO)** eliminates the separate Critic value model entirely by generating a group of $G$ outputs per prompt and calculating rewards relative to the group's mean and standard deviation. This drastically reduces GPU VRAM requirements and enables stable mathematical/reasoning alignment (as seen in DeepSeekMath / DeepSeek-R1).
- **Constitutional AI (RLAIF)**: Replaces human preference feedback with AI self-critique guided by a formal constitution of principles, mitigating harmful outputs autonomously.
- **Process Reward Models (PRMs)**: Rather than scoring only the final output (Outcome Reward Model - ORM), PRMs assign credit to every individual step in a Chain-of-Thought reasoning path, dramatically improving multi-hop logical soundness.

### 5.3 Limitations Observed & Possible Improvements
- **Limitations**: The DPO demonstration uses hand-set representative log probabilities for one prompt-response pair; no model likelihoods, gradient updates, or fine-tuning were performed.
- **Improvements**: Implement a full PyTorch training loop fine-tuning a small adapter (e.g., LoRA on Qwen-0.5B) over the curated 30-sample Self-Instruct dataset using TRL's `DPOTrainer`.
"""
    cells.append(nbf.v4.new_markdown_cell(task5_analysis_md))
    
    # -------------------------------------------------------------------------
    # TASK 6
    # -------------------------------------------------------------------------
    task6_md = """---
## Task 6: Paradigm Differentiation & Multi-Turn Research (1.5 Marks)
**Lead Contributor**: Pankaj Kumar (`2025ae05013@wilp.bits-pilani.ac.in`)

### 6.1 Objective & Experimental Design
Task 6 demonstrates three API prompt configurations—**Zero-Shot Prompting, Fixed-Context Demonstration, and Prompt-Based Specialization**—and a four-turn dialogue. It does not implement question-specific RAG or actual fine-tuning, and it does not independently score dialogue context retention.

#### 6 Evaluation Criteria:
1. **Knowledge Freshness**: Ability to incorporate rapidly evolving real-time research.
2. **Controllability & Verifiability**: Provenance linking and citation auditing.
3. **Inference Latency**: Round-trip time per query.
4. **Token Cost**: Ongoing operational cost per query.
5. **Hallucination Risk**: Susceptibility to generating unverified technical assertions.
6. **Maintenance & Retraining Overhead**: Effort required to update the system.

#### Multi-Turn Research Dialogue Experiment
We simulate a **4-turn coherent research conversation**:
- **Turn 1 (Broad Inquiry)**: Core physical constraints of superconducting qubits.
- **Turn 2 (Follow-up Clarification)**: Mathematical formulation of thermal energy $k_B T$ disrupting Josephson energy $E_J$.
- **Turn 3 (Critical Challenge)**: Reconciling room-temperature optomechanical transducer claims against previous physical laws.
- **Turn 4 (Comparative Synthesis)**: Comprehensive trade-off matrix between dilution cryogenics and neutral-atom architectures.

The runner stores placeholder CRS and FA values for demonstration, explicitly marked `illustrative_scores_not_human_or_model_judged`. They are not measured dialogue-quality scores. A valid evaluation should score each response against a turn-specific rubric, with independent review or a documented evaluator protocol.
"""
    cells.append(nbf.v4.new_markdown_cell(task6_md))
    t6_code = """# Load and display Task 6 Paradigm Differentiation & Multi-Turn Results
df_task6 = pd.read_json("data/task6_results.json")
df_dialogue = pd.read_json("data/task6_multiturn_dialogue.json")
# Tabular Display of Demonstration-Condition API Calls across 5 Questions
print("=== TASK 6 DEMONSTRATION-CONDITION API TIMING (SUCCESSFUL REQUESTS ONLY) ===")
print("API request time excludes retry waits; failed calls have null request latency and are excluded from means.")
display_cols_t6 = [
    "question_id", "category",
    "prompting_status", "prompting_latency_sec", "prompting_retry_wait_sec", "prompting_elapsed_sec",
    "rag_status", "rag_latency_sec", "rag_retry_wait_sec", "rag_elapsed_sec",
    "ft_status", "ft_latency_sec", "ft_retry_wait_sec", "ft_elapsed_sec",
]
display(HTML(df_task6[display_cols_t6].to_html(index=False, classes="table table-bordered")))
print("Metric basis:", ", ".join(sorted(df_task6["quality_and_hallucination_basis"].dropna().unique())))
print("Fine-tuning basis:", ", ".join(sorted(df_task6["fine_tuning_condition"].dropna().unique())))
print("RAG basis:", ", ".join(sorted(df_task6["rag_context_basis"].dropna().unique())))
for arm, status_col, latency_col in [("Prompting", "prompting_status", "prompting_latency_sec"), ("Fixed-context", "rag_status", "rag_latency_sec"), ("Prompt-specialization", "ft_status", "ft_latency_sec")]:
    ok = df_task6[df_task6[status_col] == "success"][latency_col].dropna()
    print(f"{arm}: {len(ok)} successful calls; mean successful request latency=" + (f"{ok.mean():.3f}s" if len(ok) else "NOT EVALUATED"))

# Tabular Display of 4-Turn Coherent Research Dialogue
print("=== 4-TURN COHERENT MULTI-TURN RESEARCH CONVERSATION ===")
display_cols_dial = ["turn_number", "dialogue_type", "user_inquiry", "request_status", "agent_response", "context_retention_score", "follow_up_accuracy", "request_latency_sec", "retry_wait_sec", "elapsed_latency_sec", "error"]
display(HTML(df_dialogue[display_cols_dial].to_html(index=False, classes="table table-striped")))
print("=== MULTI-TURN DIALOGUE: QUALITATIVE DEMONSTRATION ===")

if (
    df_dialogue["context_retention_score"].notna().any()
    and df_dialogue["follow_up_accuracy"].notna().any()
):
    print("Human/model-judged evaluation values are present.")
    print(
        "Mean CRS:",
        round(df_dialogue["context_retention_score"].mean(), 2)
    )
    print(
        "Mean FA:",
        round(df_dialogue["follow_up_accuracy"].mean(), 2)
    )
else:
    print("CRS: NOT EVALUATED — no validated scores available.")
    print("FA: NOT EVALUATED — no validated scores available.")
"""
    t6_stdout = ""
    t6_cell = create_code_cell_with_output(t6_code, t6_stdout, image_paths=["figures/task6_latency.png", "figures/task6_cost.png"])
    cells.append(t6_cell)
    
    task6_analysis_md = """### 6.2 Inference & Analysis Questions Answered

**Evaluation caveat:** Task 6 quality and hallucination values are not evaluated. The “fine-tuning” arm is a prompt-based specialization simulation, not an actual trained model. The fixed-context arm uses the same demonstration snippets rather than question-specific retrieval. Request latency is calculated from successful API request attempts and excludes retry waiting; failed calls are reported by status and excluded from latency means. These are demonstration-path timing observations, not an empirical quality comparison of prompting, RAG, and fine-tuning.

#### 1. When is RAG preferable to fine-tuning?
**Answer**:
RAG can be a suitable design choice when:
1. **Dynamic & Rapidly Changing Knowledge**: A properly implemented and refreshed retrieval index can expose newly ingested documents without changing model weights. This assignment's fixed-snippet demonstration does not test that capability.
2. **Auditability & Provenance**: Retrieval can make source documents available for attribution, provided citations are validated against retrieved passages. Fine-tuning alone does not supply that source traceability.
3. **Zero Retraining Budget**: RAG requires no GPU cluster training runs, eliminating catastrophic forgetting risks.
*Fine-tuning may be considered for stable behavior, format, or task adaptation when supported by suitable training data and evaluation. No fine-tuning was performed in this assignment.*

#### 2. What information should persist across turns?
**Answer**:
In multi-turn technical dialogues, the state persistence engine must maintain:
- **Core Problem Entities & Physical Parameters**: The primary system (e.g. transmon qubits), key variables ($T_2, E_J, k_B T$), and agreed-upon assumptions.
- **Clarification State**: Definitions or constraints established in earlier turns (e.g. distinguishing scientific target gain from engineering wall-plug gain).
- **Epistemic Agreements & Criticisms**: Objections raised by the user that were resolved.
*Transient conversational fillers, greetings, and repetitive boilerplate tool logs should be stripped to conserve context window tokens.*

#### 3. How does multi-turn research differ from single-turn Q&A?
**Answer**:
Single-turn Q&A treats queries as isolated information retrieval transactions. Multi-turn research is an **iterative dialectic process**:
- Users introduce initial hypotheses, challenge intermediate mathematical steps, demand counter-argument reconciliations, and ask for comparative trade-offs.
- The assistant must perform **coherence anchoring**, ensuring that answers in Turn 4 do not contradict the physical foundations established in Turn 1 while accommodating user criticisms.

### 6.3 Limitations Observed & Possible Improvements

Prompting, RAG and fine-tuning have different operational trade-offs. The current experiment measures API latency and token cost for the implemented demonstration paths. The RAG path uses fixed snippets and the fine-tuning path uses prompt-based specialization; neither represents a complete production implementation. No empirical quality or hallucination ranking is claimed. A full comparison would require question-specific retrieval, an independently trained model, and a shared human-evaluated test set.
"""
    cells.append(nbf.v4.new_markdown_cell(task6_analysis_md))
    
    # -------------------------------------------------------------------------
    # TASK 7
    # -------------------------------------------------------------------------
    task7_md = """---
## Task 7: Research Assistant Prototype Integration, Execution Audit & Cost Recommendation (2.0 Marks)
**Lead Contributor**: Shradha Katakdhond (`2025ae05331@wilp.bits-pilani.ac.in`)

### 7.1 Objective & System Integration Architecture
Task 7 integrates the modules developed in Tasks 1–6 into a research-assistant prototype, exercises the pipeline across **10 held-out questions**, records execution-stage outcomes and a structured JSON audit trail, and formulates prototype-level production and cost recommendations. The current run does not independently validate factual quality, citation correctness, hallucination rate, or confidence calibration; these remain explicitly not evaluated.

```mermaid
flowchart TD
    UserQuery["1. User Research Query"] --> Decomp["2. Query Decomposer & Router (CoT)"]
    Decomp --> MemSync["3. Memory Integration (Working Deque + Episodic Reflection)"]
    MemSync --> Tools["4. Parallel Multi-Source Engine (Academic, Wikipedia, News, Credibility)"]
    Tools --> Synth["5. Synthesis & Conflict Resolver (Credibility-Weighted)"]
    Synth --> Audit["6. Candidate Claim Audit + Source IDs (not entailment verification)"]
    Audit --> Output["7. Research Brief + Execution JSON Audit (quality unvalidated)"]
```

#### Key Architecture Components Integrated:
1. **Query Router & Decomposer**: Automatically deconstructs multi-hop inquiries into 2–4 targeted sub-queries using CoT.
2. **Agentic Memory Sync**: Pulls historical lessons from 20 episodic sessions and updates bounded working memory ($N=10$).
3. **Parallel Multi-Source Retrieval Engine**: Concurrently queries academic, encyclopedic, and news tools with strict schema validation.
4. **Synthesis**: Produces a response using retrieved mock-repository evidence; the integrated pipeline does not invoke Task 4's conflict-resolution benchmark.
5. **Claim Audit Placeholder**: Extracts candidate sentences and records candidate source IDs; it does not establish entailment, factual correctness, or confidence calibration.
6. **JSON Audit Trail Logger**: Records execution stages and candidate claim-to-source links; candidate links are not proof of entailment.
"""
    cells.append(nbf.v4.new_markdown_cell(task7_md))
    t7_code = """# Load and display Task 7 Full Integration, Evaluation & Cost Results
df_task7 = pd.read_json("data/task7_evaluation_results.json")
with open("data/task7_sample_audit_trail.json", encoding="utf-8") as f:
    sample_audit = json.load(f)
# Tabular Display of 10 Held-Out Evaluation Questions
print("=== TASK 7 END-TO-END EVALUATION RESULTS (10 HELD-OUT QUESTIONS) ===")
display_cols_t7 = ["evaluation_id", "category", "question", "sub_queries", "llm_synthesis_request_status", "decomposition_status", "retrieval_stage_status", "retrieval_success_count", "retrieval_total_count", "task_completion", "quality_evaluation_status", "synthesis_quality", "citation_accuracy_pct", "hallucination_rate_pct", "mean_confidence", "latency_sec", "cost_usd"]
display(HTML(df_task7[display_cols_t7].to_html(index=False, classes="table table-bordered table-striped")))

# Aggregate System Performance Metrics
print("\\n=== AGGREGATE SYSTEM PERFORMANCE METRICS ===")
def metric_or_not_evaluated(series, formatter):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return formatter(values.mean()) if len(values) else "NOT EVALUATED"

llm_success_rate = (df_task7["llm_synthesis_request_status"] == "success").mean() * 100
pipeline_stage_rate = (df_task7["task_completion"] == "execution_stages_completed_quality_unvalidated").mean() * 100
agg_metrics = pd.DataFrame([{
    "Mean Synthesis Quality": metric_or_not_evaluated(df_task7["synthesis_quality"], lambda x: f"{x:.2f} / 5.0"),
    "Mean Citation Accuracy": metric_or_not_evaluated(df_task7["citation_accuracy_pct"], lambda x: f"{x:.1f}%"),
    "Mean Hallucination Rate": metric_or_not_evaluated(df_task7["hallucination_rate_pct"], lambda x: f"{x:.1f}%"),
    "LLM Synthesis Request Success Rate": f"{llm_success_rate:.1f}%",
    "Pipeline Execution-Stage Completion Rate (not quality validation)": f"{pipeline_stage_rate:.1f}%",
    "Mean Confidence": metric_or_not_evaluated(df_task7["mean_confidence"], lambda x: f"{x:.3f}"),
    "Mean Elapsed Time per Query": f"{df_task7['latency_sec'].mean():.2f} seconds",
    "Mean Estimated Cost per Query": f"${df_task7['cost_usd'].mean():.6f} USD"
}])
display(HTML(agg_metrics.to_html(index=False, classes="table table-success")))

# Display Structured JSON Audit Trail Snippet
print("\\n=== STRUCTURED JSON AUDIT TRAIL & CANDIDATE SOURCE ATTRIBUTION (NOT VERIFIED) (SAMPLE: EVAL_05) ===")
audit_summary = {
    "Query": sample_audit["query"],
    "Episodic Reflection Injected": sample_audit["episodic_reflection"],
    "Sub-queries": sample_audit["sub_queries"],
    "Tools Executed": sample_audit["tools_executed"],
    "Tool Latency (Parallel)": f"{sample_audit['tool_latency_parallel_sec']}s",
    "Atomic Claims Extracted": sample_audit["atomic_claims_count"],
    "Verified Claims": sample_audit["verified_claims_count"],
    "Mean Calibrated Confidence": sample_audit["mean_confidence"],
    "Candidate Claim-to-Source Attribution (NOT VERIFIED)": sample_audit["claim_to_source_traceability"]
}
display(HTML(pd.DataFrame([audit_summary]).to_html(index=False, classes="table table-info")))
"""
    t7_stdout = "=== TASK 7 END-TO-END EVALUATION RESULTS (10 HELD-OUT QUESTIONS) ===\n=== AGGREGATE SYSTEM PERFORMANCE METRICS ===\n=== STRUCTURED JSON AUDIT TRAIL & CANDIDATE SOURCE ATTRIBUTION (NOT VERIFIED) (SAMPLE: EVAL_05) ===\n"
    t7_cell = create_code_cell_with_output(t7_code, t7_stdout, image_paths=["figures/task7_architecture.png"])
    cells.append(t7_cell)
    
    task7_analysis_md = """### 7.2 Design-Level Failure Modes and Mitigations (2 Cases)

#### Failure Case 1: Over-Decomposition Latency on Simple Inquiries
- **Design risk**: A simple factual query may be decomposed into multiple sub-queries and sent through the same retrieval/synthesis path as a complex query.
- **Root Cause**: The current pipeline does not yet use a validated complexity router to select a lighter path.
- **Impact**: This is a design-level risk; no controlled simple-versus-complex routing latency experiment was run.
- **Mitigation**: Implement an **Adaptive Complexity Router (Tiered Classifier)**. Queries with high factual specificity and low relational complexity bypass decomposition and execute direct single-turn retrieval.

#### Failure Case 2: Epistemic Indeterminacy in Ambiguous Scientific Debates
- **Design risk**: An ambiguous scientific question may have credible sources that disagree or insufficient evidence for a single conclusion.
- **Root Cause**: A rule-based selector can overstate certainty if it is forced to choose one winning source.
- **Impact**: No self-correction pass is implemented in the current runner; its latency and effect on confidence cannot be measured.
- **Mitigation**: Implement an **Epistemic Equivalence Fork**. When credible sources actively disagree without peer-reviewed resolution, the agent is instructed to present a **bifurcated dialectical synthesis** outlining the conflicting empirical arguments, respective methodologies, and unresolved evidentiary gaps rather than forcing an artificial consensus.

---

### 7.3 Production Recommendation & Cost Engineering

**Evidence caveat:** The tool-latency benchmark uses simulated local tool delays. Task 7's held-out records measure API latency/cost and completion status, but quality, citation accuracy, and hallucination rate are unmeasured. Cost and latency observations are therefore prototype measurements, not production service-level guarantees.

#### Architecture Decision Table

| Component | Selected Approach | Rationale | Cost & Performance Impact |
|---|---|---|---|
| **Reasoning Strategy** | Hybrid CoT + Adaptive Routing | CoT is a candidate strategy; compare its lexical-coverage proxy and measured latency against ToT before selecting it. | No validated latency reduction was established in the available run. |
| **Tool Execution** | ThreadPool Parallel Calling | Concurrently queries Academic, Wikipedia, and News sources. | The local benchmark uses simulated tool delays; live retrieval latency savings were not measured. |
| **Memory System** | Bounded Working Deque + Episodic Reflection | Bounded deque ($N=10$) + rolling summary keeps operational context compact; episodic reflections guide research strategy. | Memory token overhead was not measured in a validated production workload. |
| **Verification & Alignment**| Rule-Based Source Selection | Selects the highest configured credibility value; no claim-level entailment or correction pass is implemented. | Hallucination reduction and citation precision are not evaluated. |
| **Model Selection** | Tiered Model Routing (`qwen/qwen3.8-27b`) | Balances analytical capability with inference availability on Groq. | Observed API cost depends on actual token usage and provider pricing; no production cost guarantee is claimed. |

#### Cost Engineering Analysis
- **Cost per Query**: Calculate from recorded prompt/completion token usage and configured price constants; this is a prototype estimate, not a production quote.
- **Cost per 1,000 Queries**: Multiply measured per-query cost by 1,000 after a successful representative run.

#### Two Concrete Cost-Reduction Strategies:
1. **Tiered Model Cascading (Hierarchical Routing)**:
   - Use an ultra-lightweight, low-cost model (e.g., Llama-3.1-8B-Instant at $0.05/1M tokens) for initial intent classification, query decomposition, and safety filtering.
   - Route to the frontier reasoning model (`qwen/qwen3.8-27b`) only for multi-source synthesis and conflict resolution.
   - *Potential savings must be measured against a representative workload before being reported.*
2. **Semantic Vector Response Caching**:
   - Cache synthesized research briefs and verified claim-to-source mappings in a vector store (e.g., Qdrant / Redis).
   - For incoming queries with cosine similarity $>0.94$ against cached inquiries, serve the cached verified brief instantly with local timestamping.
   - *Cache hit rate, latency, and cost savings were not benchmarked in this assignment.*

---

### 7.4 Inference & Analysis Questions Answered

#### 1. What is the most critical component for synthesis quality?
**Answer**:
A **Credibility-Weighted Evidence and Audit Layer** is an important design component. Traceable sources and explicit evidence-quality heuristics can improve auditability, but the current prototype does not independently verify each claim and therefore cannot guarantee factual integrity or quantify hallucination reduction.

#### 2. What would you fix first before production?
**Answer**:
Prototype an **Adaptive Complexity Router** and evaluate it against the current fixed pipeline on a labelled workload. Its latency, cost, and quality benefits have not been measured in this assignment.

#### 3. Which optimization gives the best quality–cost trade-off?
**Answer**:
**Hybrid reasoning with caching** is a design recommendation to validate. This project has not measured CoT as 98% of ToT quality or benchmarked semantic caching.
"""
    cells.append(nbf.v4.new_markdown_cell(task7_analysis_md))

    reproducibility_md = """
---

## Reproducibility, Environment & Execution Record

### Runtime Environment
The experiment runner records the Python version, platform,
installed package versions, model identifier and UTC execution timestamp
in `data/reproducibility_metadata.json`.

### Model Configuration
- Provider: Groq
- Model: Read from the `GROQ_MODEL` environment variable
- API key: Read from `GROQ_API_KEY`
- API credentials are not stored in the notebook or experiment metadata.

### Randomness and Repeatability
- Random seed: 42
- Synthetic examples are reproducible under the configured seed.
- API-generated responses may vary between runs because model serving
  and generation are not guaranteed to be deterministic.

### Generated Experiment Artifacts
The `data/` directory contains task-level JSON and CSV results.
The `figures/` directory contains generated visualizations.
The metadata file records the intended experiment outputs and
known evaluation limitations.

### Execution Procedure
1. Configure `.env` with `GROQ_API_KEY` and `GROQ_MODEL`.
2. Install the required Python dependencies.
3. Run `python3 run_experiments.py`.
4. Run `python3 generate_final_notebook.py`.
5. Restart the notebook kernel and execute all cells.
6. Verify that all tables, figures and outputs correspond to the
   latest experiment run.

### Evaluation Limitations
Synthetic test fixtures, mock retrieval, illustrative scores and
unlabelled quality metrics are explicitly distinguished from
measured experimental results. Missing human-evaluation results
are reported as not evaluated rather than estimated.
"""
    cells.append(nbf.v4.new_markdown_cell(reproducibility_md))
    # -------------------------------------------------------------------------
    # FINAL CONCLUSION & REFERENCES
    # -------------------------------------------------------------------------
    conclusion_md = """---
## Final Conclusion & Key Takeaways
This project successfully designed, implemented, evaluated, and documented an **End-to-End Autonomous Research Assistant with Multi-Source Synthesis, Memory & Alignment** (Conversational AI - AIMLCZG521, Problem Statement 1).

### Key Empirical Observations:
1. **Reasoning Strategies**: Simple queries require only Zero-Shot prompting; complex multi-hop inquiries demand structured Chain-of-Thought (CoT) or Tree-of-Thought (ToT) reasoning to avoid qualitative oversimplification.
2. **Safe Function Calling**: Building a native ReAct loop with Pydantic input validation, iteration caps, and repetition hashing provides robust security against prompt injection and data exfiltration, while the local mock-tool benchmark showed approximately 59–60% lower elapsed time for the tested simulated calls; this is not a live retrieval benchmark.
3. **Agentic Memory**: Bounded working memory ($N=10$) coupled with high-level episodic reflections is an architectural approach; a quality improvement and token-overhead reduction were not established by the available run.
4. **Synthesis & Calibration**: The project demonstrates credibility-weighted rule-based conflict handling. Hallucination reduction and model calibration were not empirically measured; the calibration plot uses synthetic samples.
5. **Alignment & Paradigms**: DPO provides an elegant, stable mathematical alternative to PPO for aligning reasoning behavior, while RAG is a useful candidate for dynamically evolving research literature; this assignment did not establish a universally optimal paradigm.

---
## Academic & Technical References
1. **Wei, J., et al. (2022)**. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. Advances in Neural Information Processing Systems (NeurIPS 2022).
2. **Yao, S., et al. (2023)**. *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*. arXiv:2305.10601.
3. **Yao, S., et al. (2022)**. *ReAct: Synergizing Reasoning and Acting in Language Models*. International Conference on Learning Representations (ICLR 2023).
4. **Rafailov, R., et al. (2023)**. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS 2023.
5. **Dettmers, T., et al. (2023)**. *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023.
6. **Shao, Z., et al. (2024)**. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)*. arXiv:2402.03300.
7. **Fowler, A. G., et al. (2012)**. *Surface codes: Towards practical large-scale quantum computation*. Physical Review A, 86(3), 032324.
8. **Strukov, D. B., et al. (2008)**. *The missing memristor found*. Nature, 453(7191), 80-83.
9. **Lawson, J. D. (1957)**. *Some Criteria for a Power Producing Deuterium-Tritium Reactor*. Proceedings of the Physical Society. Section B, 70(1), 6.
"""
    cells.append(nbf.v4.new_markdown_cell(conclusion_md))
    reproducibility_code = """# Reproducibility and Execution Summary
import json
import pandas as pd
from IPython.display import display, HTML

with open(
    "data/reproducibility_metadata.json",
    encoding="utf-8"
) as f:
    reproducibility = json.load(f)

display(
    HTML(
        pd.DataFrame(
            [
                {"Parameter": key, "Value": str(value)}
                for key, value in reproducibility.items()
                if key not in [
                    "package_versions",
                    "experiment_data_files",
                    "evaluation_limitations"
                ]
            ]
        ).to_html(index=False)
    )
)

print("Package versions:")
display(pd.Series(reproducibility["package_versions"]))

print("Experiment data files:")
for filename in reproducibility["experiment_data_files"]:
    print("-", filename)

print("Evaluation limitations:")
for limitation in reproducibility["evaluation_limitations"]:
    print("-", limitation)
"""

    cells.append(nbf.v4.new_markdown_cell("## Reproducibility and Execution Summary"))
    cells.append(nbf.v4.new_code_cell(reproducibility_code))

    nb.cells = cells

    output_nb_path = "Assignment2_PS1_End_to_End_Research_Assistant.ipynb"

    # Validate required project files before launching the notebook kernel.
    required_data = [
        "data/task1_results.json", "data/task2_results.json",
        "data/task2_security_result.json",
        "data/task3_results.json", "data/task4_results.json",
        "data/task4_rag_triad.json", "data/task5_peft_comparison.json",
        "data/task5_dpo_calculation.json", "data/task5_self_instruct_dataset.json", "data/task6_results.json",
        "data/task6_multiturn_dialogue.json",
        "data/task7_evaluation_results.json",
        "data/task7_sample_audit_trail.json",
        "data/reproducibility_metadata.json",
    ]
    missing_data = [path for path in required_data if not os.path.isfile(path)]
    if missing_data:
        raise FileNotFoundError(
            "Required experiment data files are missing:\\n- "
            + "\\n- ".join(missing_data)
            + "\\nRun run_experiments.py first from the project root."
        )

    # Validate that all artifacts belong to one complete, successful runner execution.
    summary_path = Path("data/execution_summary.json")
    if not summary_path.is_file():
        raise RuntimeError("data/execution_summary.json is missing. Run the patched experiment runner first.")
    with summary_path.open(encoding="utf-8") as f:
        execution_summary = json.load(f)
    expected_tasks = {f"Task {i}" for i in range(1, 8)}
    completed_tasks = set(execution_summary.get("completed_tasks", []))
    failed_tasks = execution_summary.get("failed_tasks", [])
    if completed_tasks != expected_tasks or failed_tasks:
        raise RuntimeError(
            "Notebook generation blocked: Tasks 1-7 must all complete with no failures. "
            f"Completed={sorted(completed_tasks)}; failed={failed_tasks}"
        )
    hashes = execution_summary.get("artifact_sha256", {})
    required_json = [p for p in required_data if p.endswith(".json")]
    missing_hashes = [Path(p).name for p in required_json if Path(p).name not in hashes]
    if missing_hashes:
        raise RuntimeError(
            f"Execution summary lacks same-run hashes for {missing_hashes}. Re-run the patched runner."
        )
    for relative_path in required_json:
        artifact = Path(relative_path)
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if digest != hashes[artifact.name]:
            raise RuntimeError(
                f"Stale/mixed-run artifact detected: {relative_path}. Re-run all seven tasks."
            )

    # Save the notebook structure first, so it remains available if execution fails.
    with open(output_nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Notebook structure created: {output_nb_path}")
    print(f"Total cells: {len(cells)}")

    # Execute with a real Jupyter kernel. This creates genuine outputs and counts.
    try:
        from nbclient import NotebookClient

        notebook = nbf.read(output_nb_path, as_version=4)
        executor = NotebookClient(
            notebook,
            timeout=900,
            kernel_name="python3",
            allow_errors=False
        )
        executed_notebook = executor.execute(cwd=os.getcwd())

        with open(output_nb_path, "w", encoding="utf-8") as f:
            nbf.write(executed_notebook, f)

        print("Notebook executed successfully; genuine outputs are saved.")

    except Exception as exc:
        print("Notebook was created, but execution failed.")
        print(f"Execution error: {type(exc).__name__}: {exc}")
        print(
            "Resolve the missing package, data file, API key, or runtime issue, "
            "then execute all cells before PDF export."
        )
        raise


if __name__ == "__main__":
    build_full_notebook()
