"""HuggingFace Spaces (ZeroGPU) entrypoint.

Loads the merged fine-tuned model on a ZeroGPU H200 slot and serves a Gradio UI.
Stub — implemented in Weekend 5.
"""

from __future__ import annotations

# Will use `@spaces.GPU` decorator on the inference function so the GPU is only
# acquired during the actual generate call.
