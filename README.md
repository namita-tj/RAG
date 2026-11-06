![CI](https://github.com/namita-tj/RAG/actions/workflows/ci.yml/badge.svg)

# Table RAG: getting exact numbers out of technical documents

Most RAG pipelines are built for text. You split a PDF into chunks, embed them, retrieve a few and let a
language model answer. That works fine for paragraphs. It works much less well for tables, and in technical
documents the important stuff is usually in a table: operating limits, test parameters, tolerances,
part numbers. For these, "close enough" is wrong. If a datasheet says 8 bar and the system says 10 bar,
that's not a small mistake.

So the question I wanted to answer with this project is: **how much do tables get damaged on the way into a
RAG system, and can a table-aware pipeline do better, with numbers to show it?**

This repo has my pipeline, four common approaches to compare it against, two benchmarks and the scripts to
reproduce everything.

---

## What goes wrong with tables today

From reading the recent work and from trying the usual tools, these are the problems that kept coming up:

1. **Tables that continue on the next page fall apart.** The second half becomes a separate piece, often without
   the header row, so "1085 - 1286" no longer says what it is.
2. **Chunking cuts through tables.** Values end up in a different chunk from their column names, or a row
   gets split in half.
3. **Rows lose their context.** "Stage 5" exists in every section of a datasheet. A row without its section and
   table title is ambiguous, and retrievers mix up these near-identical siblings a lot
   ([Semantic-Answerability Gap, 2026](https://arxiv.org/pdf/2607.17742)).
4. **Numbers are treated as text.** "Which stage covers 1450 rpm?" has no 1450 anywhere in the document, only
   "1200 - 1600". Counting rows or finding the maximum needs the whole table, but retrieval only returns a
   few chunks ([TableRAG, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.710.pdf)).
5. **You can't check the answer.** The model gives you a number, but not which cell it came from.

Better parsers help with the first two (Docling's TableFormer model, for example), but they still hand back
markdown that gets chunked like text. Benchmarks on text-and-table documents show that structure mismatch
causes most retrieval failures ([From BM25 to Corrective RAG, 2026](https://arxiv.org/pdf/2604.01733)), and that
even good systems stay far below the score they reach when given the right context
([T²-RAGBench, 2026](https://arxiv.org/abs/2506.12071)).

## What I built

The idea is simple: **keep the table as a table the whole way through, and only let the system give an
answer it can point to.**

- **Ingestion that keeps every cell** (`src/tablerag/ingest/structured.py`)
  - Running headers, footers and the letterhead table repeated on every page are removed.
  - Tables continued on the next page are joined back together, whether or not the header is repeated.
    A new section heading or a new table caption stops the joining, so two similar tables aren't glued
    together.
  - Every row is stored as a record (column name → value), with its page, its section heading and its
    table caption.
- **Three kinds of chunks:** one per row (with its full context), one per small table, and a short profile per
  table (columns, row count, value ranges). Ranges like "900 - 1500" also get min and max fields, and check
  marks become "yes".
- **Exact answers without the language model** (`src/tablerag/structured_ops.py`). Range lookups, row counts and
  highest or lowest values are computed from the stored rows, with unit conversion. When a question doesn't
  clearly match one table and one operation, this step does nothing and the normal path answers instead.
  It never guesses.
- **Cite and verify** (`src/tablerag/answer.py`). The model has to answer with the id of the row it read the
  value from. The answer is only kept if that row really contains the value. Otherwise the system says
  "NOT FOUND", which you can check, instead of a number that might be wrong.

## How I compare it

Every method gets the same retriever, the same reader model, the same prompt and the same amount of retrieved
text. Only the ingestion (and, for my method, the exact path and the verification) changes.

| Method | What it is |
| --- | --- |
| Fixed-size text chunks | page text cut into 1000-character chunks with 200 overlap (LangChain style, and what this repo used to do) |
| Per-page markdown tables | text plus each table as markdown, per page (what layout-aware parsers give you) |
| Camelot | Camelot's lattice table extraction plus text chunks |
| Docling | IBM's Docling with its recommended HybridChunker (GPU run only) |
| Mine, plus ablations | my pipeline, and versions with single ideas switched off |

**Benchmarks**

- **SpecQA**: 8 synthetic product datasheets (fictional pumps, 6 models each, 13–18 pages) with tables split
  across pages in two layouts, and 445 questions with exact answers: single values, ranges, yes/no flags,
  range lookups, counts, maximums and a second small table. The generator writes the PDFs and the answers
  from the same data, so the ground truth is exact. Questions come in several phrasings so the exact path
  can't just memorise one sentence pattern.
- **T²-RAGBench**: public financial questions (FinQA and TAT-DQA, CC-BY-4.0). The contexts come as markdown, so I
  render them back into PDFs, which makes every method go through PDF ingestion.

**Metrics**

- **Ingestion:** share of table rows kept whole in one chunk, and whole *with* their section named.
- **Retrieval:** whether the evidence row is inside the first 1000 / 2000 / 4000 characters retrieved. I use a
  character budget instead of top-k because five 1000-character chunks hold far more than five rows.
- **Answers:** correct, **wrong** and refused, and precision (correct out of answered). For this kind of document the
  wrong rate is the number I care about most.

## Results

### CPU run (keyword search, no language model)

This one runs in CI on every push (`results/cpu/summary.md`).

| Method | Rows kept whole with their section | Evidence found within 2000 chars | ... for rows on continuation pages |
| --- | ---: | ---: | ---: |
| Fixed-size text chunks | 66.9% | 69.7% | 23.8% |
| Per-page markdown tables | 0.0% | 49.8% | 66.7% |
| Camelot | 0.0% | 18.6% | 23.8% |
| **Mine** | **100.0%** | **92.1%** | **88.1%** |
| Mine without row context | 0.0% | 13.6% | 19.8% |
| Mine without page stitching | 98.7% | 89.3% | 88.1% |

The exact path took 192 of the 445 questions (all the range lookups, counts and maximums) and got all 192
right. It handed every question that names a specific row to the reader, which is what it should do.

What surprised me: **row context matters more than anything else.** Without the section name on each row, my
own pipeline drops from 92% to 14%, worse than plain text chunks. All the markdown-based methods have the
same problem: their table chunks don't say which section they belong to.

### GPU run (hybrid search, Qwen2.5-3B reader, Docling, T²-RAGBench)

*Coming next.* The notebook is `notebooks/run_benchmark.ipynb`.

## Be honest about what this shows

- SpecQA is synthetic, and I wrote both the generator and the pipeline, so a high score means my pipeline handles
  the layouts I modelled. It does not prove it works on every real document. That's why T²-RAGBench is in here.
- Tables are found with pdfplumber's ruled-line detection, so **borderless tables and scanned pages are out of
  scope** for my front end. That's where a layout model like Docling's TableFormer is better. The structure
  layer on top doesn't care where the table came from, so combining the two is the obvious next step.
- The exact path only knows a few operations (range lookup, count, max/min). Everything else goes to the
  reader.
- Verification checks that the answer is printed in the cited row. It can't check calculated answers, so on
  T²-RAGBench I report my method both with and without it.

## Try it

```bash
pip install -e ".[dev]"
pytest                                   # 39 tests
python -m tablerag.bench.run             # CPU benchmark, about a minute
python -m tablerag.bench.synthetic_docs example/   # look at one of the generated datasheets
```

The full run with Docling, hybrid search and the reader model:
`pip install -e ".[gpu,baselines]"`, then the command in `notebooks/run_benchmark.ipynb` (made for a free Colab T4).

## What's in here

```
src/tablerag/
  pdf.py               reading pages: text blocks and ruled tables in order, page furniture, continuations
  normalize.py         exact parsing of numbers, ranges and units (returns nothing rather than guessing)
  ingest/              the methods: baselines.py, camelot_ingest.py, docling_ingest.py, structured.py (mine)
  retrieve.py          BM25, dense and hybrid search, shared by all methods
  structured_ops.py    exact answers from stored rows
  answer.py            the reader prompt, citations and verification
  bench/               synthetic datasheets, SpecQA, T²-RAGBench loader, metrics, the benchmark runner
tests/                 unit tests plus an end-to-end benchmark run
results/cpu/           the CPU results above
```

## Sources

- [T²-RAGBench: Text-and-Table Benchmark for Evaluating RAG](https://arxiv.org/abs/2506.12071)
- [From BM25 to Corrective RAG: Benchmarking Retrieval Strategies for Text-and-Table Documents](https://arxiv.org/pdf/2604.01733)
- [TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning](https://aclanthology.org/2025.emnlp-main.710.pdf)
- [Semantically Similar, Logically Distinct: Diagnosing the Semantic-Answerability Gap in Table RAG](https://arxiv.org/pdf/2607.17742)
- [Structure-Aware Chunking for Tabular Data in RAG](https://arxiv.org/html/2605.00318v1)
- [Docling](https://github.com/docling-project/docling)
