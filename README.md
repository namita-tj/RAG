![CI](https://github.com/namita-tj/RAG/actions/workflows/ci.yml/badge.svg)

# Table RAG: getting exact numbers out of technical documents

RAG works well on plain text, but tables in technical and scientific documents still break it. Tables get split
across pages, chunking cuts rows away from their headers, and a row like "Stage 5" means nothing without its
section. The result is an answer that is close but wrong, and in a datasheet 8 bar instead of 10 bar is not a
small mistake.

This project is a table-aware RAG pipeline, benchmarked against four common approaches including Docling.
The idea: **be deterministic where precision matters, and use a language model only where it's needed.** A
number should either be exact and traceable to its cell, or the system should say it doesn't know.

## How it works

1. **Keep tables intact.** Tables split across pages are stitched back together, and page headers and footers are
   removed. Every row is stored as a record with its page, section heading and table caption.
2. **Retrieve rows, not fragments.** Each row becomes its own chunk with that context, next to whole-table and
   text chunks. Numbers and units in the question are written the way documents print them ("1,450 revolutions
   per minute" becomes "1450 rpm"). Search is hybrid, BM25 plus `bge-small` embeddings, with an optional
   cross-encoder reranker.
3. **Compute what can be computed.** Counts (also filtered: "how many stages allow intermittent duty"), totals,
   means, maximums, "which row has the highest value" and "which range contains this value" are answered from
   the stored table with unit conversion, without the language model.
4. **Check every answer.** The model's answer is only kept if the value is printed in the retrieved evidence. If
   the answer needs a calculation, the model writes the formula and the code computes it, and only if every
   number in the formula is printed in the evidence. Otherwise the answer is "NOT FOUND".

## Results

Same retriever, same reader (Qwen2.5-3B), same 2,000 characters of context for every method; only the pipeline
differs. Run on a free Colab T4.

| Benchmark | Method | Correct | Wrong | Precision when answered |
| --- | --- | ---: | ---: | ---: |
| **SpecQA** (synthetic datasheets, 445 questions) | Docling | 37.1% | 52.8% | 41.2% |
| | Fixed-size chunks | 32.1% | 60.2% | 34.8% |
| | **Mine** | **94.8%** | **3.1%** | **96.8%** |
| **T²-RAGBench** (financial reports, 730 questions) | Docling | 21.5% | 61.9% | 25.8% |
| | Fixed-size chunks | 19.6% | 65.5% | 23.0% |
| | **Mine** | 14.5% | **18.5%** | **44.0%** |

- On datasheets, the pipeline gets 2.5 times as many answers right as Docling, and wrong answers drop from 53% to 3%.
- On financial reports, wrong answers drop from about 62% to 18.5%. The cost is refusals: most of these questions
  need a calculation whose result is printed nowhere, and the check can't verify that yet.
- The biggest single factor is **row context**. Without the section name on each row, finding the right row
  within 1,000 characters drops from 81% to 0%.

All methods, ablations, metrics and the limits of this comparison: [docs/benchmark.md](docs/benchmark.md).

## Using it

```bash
pip install -e .
tablerag index ./pdfs --db index.db
tablerag ask "Which stage of the PX-100 covers 1450 rpm?" --db index.db
```

```
Answer: Stage 7   (structured)
Because: 1450 rpm lies in Stage 7 of 4.1 PX-100 - Compact pump
Source: datasheet_7, page 4, Table 4.1-1: Operating points (4.1 PX-100 - Compact pump)
  Stage: Stage 7; Speed [rpm]: 1380 - 1963; Duty: Intermittent; Max. pressure [bar]: 4
```

- The index is one SQLite file. Indexing again only reads new or changed PDFs.
- Without a reader model, counts, ranges and maximums are still answered; other questions show the best evidence.
- `--reader Qwen/Qwen2.5-3B-Instruct` runs the full pipeline, and `--embed-model BAAI/bge-small-en-v1.5` at
  indexing time enables `--retriever hybrid`. Both need `pip install -e ".[gpu]"`.

How the pipeline and the storage are designed, and why: [docs/design.md](docs/design.md).

## Development

```bash
pip install -e ".[dev]"
pytest                          # 71 tests
python -m tablerag.bench.run    # CPU benchmark, about a minute
```

GitHub Actions does the rest:

| Workflow | When | What it does |
| --- | --- | --- |
| CI | every push and pull request | lint, tests, CPU benchmark, **benchmark gate**, the CLI end to end, links in the docs, Docker build |
| Weekly dependency check | Mondays | the same checks against the newest dependency versions, to catch upstream changes in PDF libraries |
| Release | version tag `v*` | tests and gate, then a GitHub Release with the package and benchmark results, and a Docker image on `ghcr.io` |
| CodeQL | pull requests, weekly | security scan of the code |
| Dependabot | monthly | update pull requests, which have to pass the gate |

The **benchmark gate** fails a run if any number of my method gets worse than the committed baseline in
`results/cpu/`, and shows the comparison table on the run's summary page. The full GPU benchmark is
`notebooks/run_benchmark.ipynb`. To release: set the version in `pyproject.toml`, then
`git tag v0.2.0 && git push origin v0.2.0`.

## Limits

- Only digital PDFs with ruled tables.
- SpecQA is synthetic and was written alongside the pipeline. Each result is one run, without error bars.
- The GPU results above are from before the filtered counts, totals, calculations and query clean-up; a rerun of
  my method updates them.
- The embedding model, the BM25 weight and the 2,000-character context are sensible defaults, not tuned values.
  The retrieval sweep (`python -m tablerag.bench.sweep`, details in [docs/benchmark.md](docs/benchmark.md)) tunes
  the first two on one half of the documents and reports on the other half.

## Future work

**Reading more kinds of documents**
- Borderless tables, found by how their words line up in columns.
- Scanned pages and tables that are images: OpenCV for the grid, Tesseract for the text, and refusal when the OCR
  is unsure of a number.
- Symbols, superscripts and subscripts (m/s², x²) from font size and position; full formulas to LaTeX.
- Merged cells and multi-level headers, rotated pages and multi-column layouts.
- Docling's table model as a fallback reader, with this structure layer on top.

**Answering**
- Tuning stage 2: context budget (1,000 / 2,000 / 4,000 characters) and reader models of other families and sizes
  (Llama 3.2 3B, Phi-3.5-mini, Qwen2.5 7B in 4-bit), with the time per question next to the accuracy.
- Exact operations as SQL queries over the stored cells, with a test that they match the Python version.
- A risk-coverage curve: precision when answering 50%, 80% or all questions, instead of one cut-off.

**Evaluation**
- A held-out benchmark on real documents from several fields, and error bars over several runs.
- One embedding run through the saved index with the real model (the tests use a stand-in).

**Using it**
- An HTTP API and a small web demo on top of the index.

## Project layout

```
src/tablerag/   pdf reading, ingestion methods, retrieval, exact operations, answering, store, CLI
  bench/        benchmark generators, runner, metrics, merge and gate
tests/          unit tests and an end-to-end benchmark run
results/        CPU and GPU benchmark results
docs/           design and benchmark details
```
