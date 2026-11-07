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
   text chunks. Search is hybrid: BM25 plus `bge-small` embeddings.
3. **Compute what can be computed.** Counts, maximums and "which range contains this value" are answered from the
   stored table with unit conversion, without the language model.
4. **Check every answer.** The model's answer is only kept if the value is printed in the retrieved evidence.
   Otherwise the answer is "NOT FOUND".

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
pytest                          # 54 tests
python -m tablerag.bench.run    # CPU benchmark, about a minute
```

CI runs linting, the tests and the CPU benchmark on every push. A **benchmark gate** fails the run if any number
of my method gets worse than the committed baseline in `results/cpu/`, and shows the comparison table on the
run's summary page. The full GPU benchmark is `notebooks/run_benchmark.ipynb`.

## Limits and next steps

- Only digital PDFs with ruled tables. Borderless tables, scanned pages and formulas are not handled yet.
- Calculated answers can't be checked yet. Next: let the model write the formula over specific cells and let the
  code calculate it.
- SpecQA is synthetic and was written alongside the pipeline. A held-out benchmark on real documents is next.

## Project layout

```
src/tablerag/   pdf reading, ingestion methods, retrieval, exact operations, answering, store, CLI
  bench/        benchmark generators, runner, metrics, merge and gate
tests/          unit tests and an end-to-end benchmark run
results/        CPU and GPU benchmark results
docs/           design and benchmark details
```
