# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.9.1",
#     "torchvision==0.24.1",
#     "transformers>=5.18",
#     "sentence-transformers>=5.1",
#     "accelerate",
#     "pillow",
#     "numpy<3",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# torchvision = { index = "pytorch-cu128" }
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
#
# omni_worker.py -- the shared-space encoder, in an environment of its own
#
# EmbeddingGemma 2 puts text, pictures, sound and video in one 768-number
# space, which is what lets a sound find a picture. It was published against
# transformers 5.18. Arcana itself is held at transformers 4.x, because version
# 5 changed what the CLIP calls return and nothing in Alchemy or the diffusion
# methods has been run against it. So this file is not part of the app's
# environment. `uv run --script` builds it one from the block above, and the
# app talks to it over a pipe (arcana/omni.py).
#
# It imports nothing from Arcana. The app decodes every file with its own
# readers and hands over arrays: a picture or the frames of a video section as
# uint8, sound as float32 at 16 kHz. That keeps one decoder for each kind of
# media, and keeps this script to the model alone.
#
# Protocol, one JSON object per line.
#   in   {"id": 7, "items": [{"kind": "text", "text": "...", "role": "query"},
#                            {"kind": "image", "npy": "<path>"},
#                            {"kind": "video", "npy": "<path>"},
#                            {"kind": "audio", "npy": "<path>"}], "out": "<path>.npy"}
#   out  {"id": 7, "ok": true, "n": 4, "dim": 768}        vectors are written to `out`
#        {"id": 7, "ok": false, "error": "..."}
#   The first line it writes is {"ready": true, "model": ..., "device": ..., "dim": ...}.
#   {"quit": true} ends it.

import json
import sys
import traceback

MODEL = "google/embeddinggemma-2"


def say(obj) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main() -> int:
    # Everything the libraries print goes to stderr, so stdout carries the protocol alone.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        import numpy as np
        import torch
        from PIL import Image
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # The model card: bfloat16 or float32. float16 returns NaN without an error.
        kwargs = {"torch_dtype": torch.bfloat16} if device == "cuda" else {}
        # From disk first: asking the hub about files that are already here is
        # what a dropped connection breaks (see arcana/hub.py).
        try:
            model = SentenceTransformer(MODEL, device=device, model_kwargs=kwargs, local_files_only=True)
        except Exception:                                    # noqa: BLE001
            model = SentenceTransformer(MODEL, device=device, model_kwargs=kwargs)
        prompts = dict(getattr(model, "prompts", {}) or {})
        dim = int(model.get_sentence_embedding_dimension() or 768)
    except Exception as e:                                   # noqa: BLE001
        sys.stdout = real_stdout
        say({"ready": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})
        return 1
    sys.stdout = real_stdout
    say({"ready": True, "model": MODEL, "device": device, "dim": dim, "prompts": sorted(prompts)})

    query_prompt = next((p for p in ("SearchQuery", "search_query", "query", "Retrieval-query") if p in prompts), None)

    def text_vectors(texts, role):
        if role == "query":
            if query_prompt:
                return model.encode(texts, prompt_name=query_prompt, normalize_embeddings=True)
            return model.encode([f"task: search result | query: {t}" for t in texts], normalize_embeddings=True)
        return model.encode(texts, normalize_embeddings=True)

    def one(item):
        kind = item["kind"]
        if kind == "text":
            return text_vectors([item["text"]], item.get("role", "query"))[0]
        arr = np.load(item["npy"])
        if kind == "image":
            return model.encode({"image": Image.fromarray(arr)}, normalize_embeddings=True)
        if kind == "video":
            frames = [Image.fromarray(f) for f in arr]
            return model.encode({"video": frames}, normalize_embeddings=True)
        if kind == "audio":
            return model.encode({"audio": arr.astype(np.float32)}, normalize_embeddings=True)
        raise ValueError(f"unknown kind {kind!r}")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            say({"ok": False, "error": "not JSON"})
            continue
        if req.get("quit"):
            break
        sys.stdout = sys.stderr
        try:
            items = req["items"]
            # Texts of one role go through together; everything else one at a time,
            # since a section's frames and a picture do not batch into one tensor.
            vecs = [None] * len(items)
            for role in ("query", "document"):
                idx = [i for i, it in enumerate(items) if it["kind"] == "text" and it.get("role", "query") == role]
                if idx:
                    got = text_vectors([items[i]["text"] for i in idx], role)
                    for i, v in zip(idx, got):
                        vecs[i] = v
            for i, it in enumerate(items):
                if vecs[i] is None:
                    vecs[i] = one(it)
            out = np.asarray(np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in vecs]), np.float32)
            if not np.isfinite(out).all():
                raise ValueError("the model returned NaN; it must run in bfloat16 or float32")
            np.save(req["out"], out)
            reply = {"id": req.get("id"), "ok": True, "n": int(out.shape[0]), "dim": int(out.shape[1])}
        except Exception as e:                               # noqa: BLE001
            reply = {"id": req.get("id"), "ok": False, "error": f"{type(e).__name__}: {e}",
                     "trace": traceback.format_exc()[-1500:]}
        sys.stdout = real_stdout
        say(reply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
