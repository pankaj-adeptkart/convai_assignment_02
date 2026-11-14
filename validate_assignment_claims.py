"""Offline validation of deterministic calculations and report/code consistency.
This does NOT validate scientific facts or run the live Groq experiment.
"""
import ast, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
runner = ROOT / "run_experiments.py"
generator = ROOT / "generate_final_notebook.py"
notebook = ROOT / "Assignment2_PS1_End_to_End_Research_Assistant.ipynb"
for path in (runner, generator):
    ast.parse(path.read_text(encoding="utf-8"))
nb = json.loads(notebook.read_text(encoding="utf-8"))
assert nb.get("nbformat") == 4
assert len(nb.get("cells", [])) > 0

# Independently recompute the DPO arithmetic using the documented illustrative inputs.
beta = 0.1
log_pi_theta_yw, log_pi_ref_yw = -14.2, -18.5
log_pi_theta_yl, log_pi_ref_yl = -22.8, -19.1
rw = log_pi_theta_yw - log_pi_ref_yw
rl = log_pi_theta_yl - log_pi_ref_yl
z = beta * (rw - rl)
sigmoid = 1.0 / (1.0 + math.exp(-z))
loss = -math.log(sigmoid)
assert round(rw, 4) == 4.3
assert round(rl, 4) == -3.7
assert round(z, 4) == 0.8
assert round(sigmoid, 5) == 0.68997
assert round(loss, 5) == 0.3711

# Source/report claims: labels must disclose non-empirical and unverified stages.
R = runner.read_text(encoding="utf-8")
G = generator.read_text(encoding="utf-8")
assert "soft_wait_timeout_thread_may_continue" in R
assert "generation_source" in R and "procedural_fallback" in R
assert "training_performed" in R
assert "NOT_AUTOMATICALLY_VERIFIED" in R
assert "task6_latency.png" in R and "task6_cost.png" in R
assert "Claim Audit Placeholder" in G
assert "Synth --> Verifier" not in G
assert "not entailment verification" in G or "candidate claim" in G.lower()
assert "90\\%" not in G and ">90%" not in G
print("PASS: Python AST, notebook JSON, DPO arithmetic, and code/report limitation consistency.")
print("NOT TESTED: live Groq API, full seven-task execution, scientific correctness, citation entailment, human quality scores.")
