"""
Script to assemble the master Jupyter Notebook for Conversational AI Assignment 2 (PS1).
Embeds all markdown analysis, student details, contribution matrix, code implementations,
live outputs, and visualizations.
"""

import os
import json
import nbformat as nbf

def create_master_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    }
    cells = []
    
    # Header Cell
    title_md = """# Problem Statement 1: End-to-End Research Assistant with Multi-Source Synthesis, Memory & Alignment
## Conversational AI (AIMLCZG521) — EC-1 Assessment (Total Marks: 10)
---
### Student Details & Group Contribution Evidence

| Student Name | BITS Student ID / Email | Assigned Contribution Areas | Section Responsibilities |
|---|---|---|---|
| **Shradha Katakdhond** | `2025ae05331@wilp.bits-pilani.ac.in` | • Prompt Engineering & Multi-Step Reasoning<br>• Full Integration, Evaluation & Cost Engineering | **Task 1**: Reasoning strategy evaluation (Zero, Few, CoT, ToT, APE), metrics & trade-offs.<br>**Task 7**: Unified `ResearchAssistant` class, 10 held-out evaluation, failure root-cause analysis, JSON audit trail, production recommendation. |
| **Mallu Lakshmi** | `2025ae05117@wilp.bits-pilani.ac.in` | • Multi-Source Function Calling & Security<br>• Visualization & Reporting | **Task 2**: 5 JSON schemas, ReAct loop, parallel vs sequential benchmarks, 6 safety mechanisms, attack-defense demonstration.<br>**Cross-Cutting**: Architectural schematics, comparative charts, and PDF report compilation. |
| **Pankaj Kumar** | `2025ae05013@wilp.bits-pilani.ac.in` | • Memory, Decomposition & Context Engineering<br>• Paradigm Differentiation & Multi-Turn Research | **Task 3**: Bounded working memory, 20 synthetic episodic sessions with reflection, query decomposition, context ablation.<br>**Task 6**: Prompting vs RAG vs Fine-Tuning empirical benchmarks, decision framework, 4-turn research dialogue (CRS & FA). |
| **Aditya Reddy** | `2025ae05196@wilp.bits-pilani.ac.in` | • Multi-Source Synthesis, Calibration & RAG Triad<br>• Instruction Tuning, RLHF, PEFT & Alignment | **Task 4**: Multi-source evidence evaluation, 5 conflict scenarios & resolution, hallucination self-correction, calibration & ECE, RAG Triad.<br>**Task 5**: Self-Instruct pipeline (30 instructions), exact DPO loss calculation walkthrough, PEFT (Full FT vs LoRA vs QLoRA), PPO vs GRPO. |

---
### Assignment Metadata & Architecture Overview
- **Domain**: Advanced Science & Computing (Superconducting Quantum Processors, Neuromorphic AI & Synaptic Plasticity, Magnetic Confinement Fusion).
- **Core LLM Engine**: Live Groq Cloud API using `qwen/qwen3.8-27b`.
- **Framework Constraint**: Pure Python scratch implementation of ReAct loops, safety controls, working/episodic memory, and conflict resolution (no LangChain / pre-built agent wrappers).
"""
    cells.append(nbf.v4.new_markdown_cell(title_md))
    return nb

if __name__ == "__main__":
    print("Notebook builder template ready.")
