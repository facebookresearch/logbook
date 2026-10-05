"""Training-side code for long-audio segmentation.

``data.py`` turns the Ego4D ``annotated_manifest.json`` + audio into
SFT chat-format examples (system prompt + audio -> target segmentation
JSON), matching the exact chunking / prompt / schema conventions used by
inference (``long_audio.inference.chunk_runner``) so a fine-tuned model
is trained on the same distribution it is evaluated on.
"""
