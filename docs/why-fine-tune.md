# Why fine-tune (and why not just RAG?)

The most common mistake in 2025-26 AI engineering portfolios is reaching for
fine-tuning when RAG would do the job, or reaching for RAG when fine-tuning
would. EarningsLoRA exists in part to demonstrate the right intuition.

## When fine-tuning is the right tool

Fine-tuning is for **behaviour**, not **knowledge**:

- **Style and format learning** — bullet structure, tone, granularity, ordering.
  These are properties of the *output*, not the input.
- **Latency / cost at scale** — once trained, a small local model beats frontier
  APIs on $/request and on cold-start latency. The crossover happens around a
  few hundred requests per day.
- **Schema compliance** — predictable output structure for downstream pipelines.
  Easier to enforce by training than by prompt-engineering.
- **Domain-specific tone** — buy-side analyst register isn't something a generic
  instruction-tuned model lands on by default.

## When RAG is the right tool

RAG is for **knowledge**, not **behaviour**:

- The model needs facts that aren't in its parametric memory or its current
  context window.
- The corpus updates frequently (you can't retrain every time it changes).
- Provenance / citation matters — RAG returns sources alongside the answer.

## Why this task is fine-tuning, not RAG

The transcript is **fully in-context**. The model isn't being asked to retrieve
external facts; it's being asked to *transform* the input into a different
output style. There is nothing to retrieve. RAG would just add noise.

What we *are* asking the model to learn:

- Bullet structure (5-10 bullets, decision-relevant ordering).
- Buy-side analyst register (not a generic neutral summary).
- Faithfulness to numbers (no hallucinated figures — measured by `numeric_recall`).
- Length and density appropriate for an investment workflow.

All of these are behavioural. All are hard to nail with a system prompt. All
benefit from gradient updates on (transcript, summary) pairs.

## Why a 3B model and not 70B

Three reasons:

1. **It fits a free T4 16 GB with QLoRA.** Larger models would need either rented
   GPU or a slower distributed setup.
2. **Production economics.** A 3B local adapter is cheaper and faster per request
   than any frontier API at any meaningful volume. Demonstrating that crossover
   is half the point of the project.
3. **It's a harder, more honest setup.** Fine-tuning a 70B model and beating GPT-4
   is impressive but unreplicable on a free stack. Closing most of the gap with
   3B is a more useful real-world signal.
