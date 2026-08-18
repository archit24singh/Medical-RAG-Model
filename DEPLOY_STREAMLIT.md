# Deploying the ICD-10 RAG demo to Streamlit Community Cloud

This is the free, single-process deployment: no FastAPI, no PostgreSQL, no
separate ChromaDB service. The Streamlit app loads a **pre-built (baked)** Chroma
index committed in this repo, embeds the query and reranks in-process on CPU, and
calls **Groq** (primary) with an **OpenRouter** fallback for the answer.

## What's in the deploy

| Path | Purpose |
|------|---------|
| `streamlit_app.py` | App entry point (Streamlit "Main file"). |
| `requirements.txt` | Runtime deps, pinned to what the index was built with. |
| `chroma_index/` | Pre-built vector index (bge-small embeddings). **Committed.** |
| `backend/` | Reused retrieval code (RAG-only paths). |
| `scripts/build_index.py` | Rebuild the index if the documents change. |
| `.streamlit/secrets.toml.example` | Template for the secrets you paste in the UI. |

Runtime config is forced in `streamlit_app.py`: `RAG_ONLY_MODE=true`,
`EMBEDDING_BACKEND=sentence-transformers`, `CHROMA_MODE=persistent`,
`ENABLE_HALLUCINATION_CHECK=false` (one LLM call per question).

## Step 1 — Push to GitHub

From the repo root:

```bash
git add -A
git commit -m "Add Streamlit deploy: baked index, RAG-only mode, Groq LLM chain"
git push origin main
```

> Double-check `.env` is **not** staged (`git status` should not list it). It's
> gitignored, so your keys stay local.

## Step 2 — Create the app on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in **with GitHub**.
2. Click **Create app → Deploy a public app from GitHub**.
3. Repository: `archit24singh/Medical-RAG-Model`. Branch: `main`.
4. **Main file path:** `streamlit_app.py`.
5. Click **Deploy**. The first build takes a few minutes (it installs torch etc.).

## Step 3 — Add your secrets

In the app: **Settings → Secrets**, paste (with real values):

```toml
GROQ_API_KEY = "gsk_your_key_here"
GROQ_MODEL   = "openai/gpt-oss-120b"

# Optional fallback:
OPENAI_API_KEY  = "sk-or-v1-your_key_here"
OPENAI_MODEL    = "nvidia/nemotron-3-ultra"
OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
```

Save — the app reruns and picks them up. Ask *"What is the Z code for acquired
absence of limb?"* → it should answer **Z89**.

## Rebuilding the index (only if documents change)

```bash
BUILD_FRESH=1 python scripts/build_index.py "bucket/uploads/<your>.pdf"
git add chroma_index && git commit -m "Rebuild index" && git push
```

The app auto-redeploys on push.

## Troubleshooting

- **`torch==2.13.0+cpu` won't resolve during build.** Replace that line in
  `requirements.txt` with a nearby CPU version (e.g. `torch==2.12.0+cpu`) or just
  `torch` (larger build). Keep the `--extra-index-url` line.
- **App runs out of memory (~1 GB ceiling).** Two levers, cheapest first:
  set `RERANKER_TOP_K = "3"` (already default) and, if still tight, disable the
  CrossEncoder reranker by lowering quality — tell me and I'll add a
  `USE_RERANKER=false` switch. Also confirm `ENABLE_HALLUCINATION_CHECK=false`.
- **First question after idle is slow.** The app sleeps when unused; the first
  hit reloads models (~20–60s), then it's fast. Warm it up before a demo.
- **"No API key configured."** You haven't added `GROQ_API_KEY` in Settings →
  Secrets, or the app hasn't rerun since you did.
