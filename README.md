![CI](https://github.com/namita-tj/RAG/actions/workflows/ci.yml/badge.svg)

# Table RAG: getting exact numbers out of technical documents

Every industry has documentation, and RAG is the usual way to make it searchable: ingest the documents, store
them, retrieve the right piece and let a language model answer. For plain text that works well, and for images
there are good computer-vision pipelines. Tables are where it still falls short, especially in scientific,
technical and mathematical documents, where the results and the data live in tables. Tables come in many
layouts, they get split across pages, some are nested, and when a pipeline stores them it often loses the
structure. Then retrieval misses, and the answer is a number that is close but wrong. In a technical document,
8 bar instead of 10 bar is not a small mistake.

So I wanted to look at what the current table-RAG tools (Docling and others) do, find where they fall short,
and answer one question: **how much do tables get damaged on the way into a RAG system, and can a table-aware
pipeline do better, with numbers to show it?**

**Who it's for:** anyone who works with technical PDFs where the data is in tables: research papers with result
tables, datasheets, specifications, test reports.

This repo has my pipeline, four common approaches to compare it against, two benchmarks and the scripts to
reproduce everything.

## How I got here

I started with a computer-vision approach: detect the tables in the page image and recognise their structure.
It worked for simple tables, but every approach I tried had its weak spots, and technical tables with formulas
and symbols were the hardest. So instead of picking one tool, I first went through the open problems of the
current methods (next section) and built on those.

The idea I ended up with: **be deterministic where precision matters, and use a language model only where
flexibility is needed.** Counting rows, finding a maximum or looking up which range contains a value is
computed exactly from the stored table. Only the questions that need understanding go to the model, and its
answer has to be printed in the retrieved evidence. A number should either be exact and traceable, or the
system should say it doesn't know.

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
- **Check every answer against the evidence** (`src/tablerag/answer.py`). After the model answers, the code
  checks that the value is actually printed in a retrieved row or passage. If several rows print it, the row
  the question names wins. A yes/no answer is only kept when the row the question asks about was retrieved.
  Otherwise the system says "NOT FOUND", which you can check, instead of a number that might be wrong.

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

This one runs in CI on every push and pull request (`results/cpu/summary.md`), followed by a **benchmark gate**
(`src/tablerag/bench/gate.py`): it compares the run with the committed numbers and fails if anything about my
method gets worse, for example rows losing their section, evidence found less often, or the exact path giving
a single wrong answer. The CPU run is deterministic, so any drop counts. The comparison table appears on the
run's summary page in GitHub Actions, so a pull request shows its effect on the numbers before it's merged.
If a change is meant to move the numbers, the pull request updates `results/cpu/` too, and the diff shows it.

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

Full tables: `results/gpu/summary.md`. Every method gets the same hybrid retriever (BM25 + `bge-small`), the
same reader (Qwen2.5-3B-Instruct) and 2000 characters of retrieved text. Run on a free Colab T4.

**SpecQA: answers (445 questions)**

| Method | Correct | Wrong | Refused | Precision when answered |
| --- | ---: | ---: | ---: | ---: |
| Fixed-size text chunks | 32.1% | 60.2% | 7.6% | 34.8% |
| Per-page markdown tables | 16.0% | 73.0% | 11.0% | 17.9% |
| Camelot | 15.5% | 75.7% | 8.8% | 17.0% |
| Docling | 37.1% | 52.8% | 10.1% | 41.2% |
| Mine, ingestion only (same reader, no exact path, no check) | 63.1% | 35.3% | 1.6% | 64.2% |
| **Mine (full)** | **94.8%** | **3.1%** | **2.0%** | **96.8%** |

With the same reader and the same amount of text, my pipeline gets 2.5 times as many answers right as Docling,
and wrong answers drop from 53% to 3%. Ingestion alone already takes it from 37% to 63%. The exact path does
the rest: counts, maximums and "which stage covers X" are questions the reader gets wrong almost every time
(Docling: 23%, 6% and 1%), and the exact path gets all of them right.

Retrieval tells the same story: the evidence row is inside the first 1000 characters for 81% of questions
(Docling 66%), and for rows on continuation pages it's within 2000 characters for 94% (Docling 57%).

**T²-RAGBench: answers (730 questions on financial reports)**

| Method | Correct | Wrong | Refused | Precision when answered |
| --- | ---: | ---: | ---: | ---: |
| Fixed-size text chunks | 19.6% | 65.5% | 14.9% | 23.0% |
| Per-page markdown tables | 17.9% | 56.2% | 25.9% | 24.2% |
| Camelot | 16.6% | 52.3% | 31.1% | 24.1% |
| Docling | 21.5% | 61.9% | 16.6% | 25.8% |
| Mine without the check | 21.6% | 67.0% | 11.4% | 24.4% |
| **Mine (full)** | 14.5% | **18.5%** | 67.0% | **44.0%** |

This one is harder and closer to real life. Every method finds the right document (my retrieval is the best at
97.8% within 1000 characters, Docling 96.3%), but a 3B reader gets most answers wrong, mainly because most
FinQA questions need a calculation. Here the check is what matters: wrong answers drop from about 62-67% to
18.5% and precision nearly doubles. The price is that calculated answers (a ratio, a difference) are printed
nowhere, so the check refuses them: FinQA correct goes down to 1%. Making those checkable is the next step
(see below).

**What I learned from the ablations**

- **Row context matters most.** Without the section name on each row, retrieval within 1000 characters drops
  from 81% to 0% on SpecQA and from 97.8% to 89.3% on T²-RAGBench.
- **Stitching matters for split tables:** rows on continuation pages found within 1000 characters go from 87%
  to 75% without it.
- **My first verification design failed.** I first made the model cite the row id with every answer. The 3B
  reader then said "NOT FOUND" to most questions it could answer: 55% refused on SpecQA, 98% on T²-RAGBench,
  while the same model answered every simple value question correctly without that instruction. Moving the
  check from the prompt into code fixed it (43.6% → 94.8% correct). The old design stays in the results as
  "citing row ids" so the comparison is visible.
- **What surprised me most:** how much row context matters. I expected stitching split tables to be the big win.
  It does help (above), but a row that doesn't say which section and table it belongs to is almost useless to
  a retriever, whatever else the pipeline does right.
- **On SpecQA the check adds little.** Without it, slightly more answers are right (96.4%) and slightly more are
  wrong (3.6%). My yes/no rule is too strict: correct yes/no answers drop from 81% to 67%.

How the GPU numbers were made: one full run with every method, then a second run of only my method after
the fix to the check (`results/gpu/runs/`). `python -m tablerag.bench.merge` combines them and rescores every
method with the same rules, so a reply like "NOT_FOUND" counts as a refusal for all of them.

## Be honest about what this shows

- **Home advantage.** SpecQA is synthetic, and I wrote both the generator and the pipeline. Its question types
  (counts, maximums, range lookups) are exactly what my exact path handles. A high score means my pipeline
  handles the layouts and questions I modelled, not that it works on every real document. That's why
  T²-RAGBench is in here.
- **Every table in both benchmarks has ruled lines**, because I render them that way. That suits my table
  detector and Camelot. Real documents often have borderless tables, where Docling is stronger.
- **No held-out test set.** I changed the evidence check after seeing the results on the same questions, so the
  SpecQA numbers for the final version are a bit optimistic. The fix is a new set of documents the pipeline
  has never seen.
- **One run, no error bars.** With 445 and 730 questions, a difference of a couple of points can be noise. The big
  gaps (95% vs 37% correct) are not.
- The fairest single comparison is "mine, ingestion only" against Docling: same reader, same prompt, no exact
  path and no check. That's 63% vs 37% correct on SpecQA.
- Tables are found with pdfplumber's ruled-line detection, so **borderless tables and scanned pages are out of
  scope** for my front end. That's where a layout model like Docling's TableFormer is better. The structure
  layer on top doesn't care where the table came from, so combining the two is the obvious next step.
- The exact path only knows a few operations (range lookup, count, max/min). Everything else goes to the
  reader.
- The check only confirms that the value is printed in the retrieved evidence. It can't check calculated
  answers, so it refuses them, and it can't tell if a value printed in two rows came from the right one unless
  the question names the row.
- The reader is a 3B model. A bigger reader would score higher for every method; the comparison is about
  what each pipeline gives the same reader.

## What's next

- **Checkable calculations.** Let the model write the formula over specific cells (for example
  `(revenue 2019 - revenue 2018) / revenue 2018`) and have the code calculate it. Then calculated answers can be
  checked too, which is exactly where FinQA fails today.
- **A risk-coverage curve** instead of one cut-off: precision when answering 50%, 80% or all questions.
- **Docling's table model as the front end** for borderless and scanned tables, with my structure layer on top.
- **Rows in a real database.** Store the rows in SQLite or DuckDB and run the exact operations as SQL, so the
  deterministic part scales to large document collections.
- **Formulas, symbols and nested tables**, which the current front end doesn't handle.
- **A held-out benchmark on real documents** from different fields, to see whether it generalises.

## Looking back

If I started again, I would begin with the evaluation, not the pipeline. First the scope: what data I have, what
I benchmark against, how I measure performance and how I check whether it generalises across industries and
document types, and whether it scales. Then build. I would also test much earlier which parts need a
language model at all. In this project the deterministic parts (the exact table operations and the check done
in code) moved the accuracy most, and the one time I relied on the model to do bookkeeping, citing row ids,
it failed.

## Try it

```bash
pip install -e ".[dev]"
pytest                                   # 47 tests
python -m tablerag.bench.run             # CPU benchmark, about a minute
python -m tablerag.bench.gate --current benchmarks/results/results.json   # compare with results/cpu
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
  answer.py            the reader prompt and the check against the evidence
  bench/               synthetic datasheets, SpecQA, T²-RAGBench loader, metrics, the benchmark runner,
                       merge.py to combine runs, gate.py to catch regressions in CI
tests/                 unit tests plus an end-to-end benchmark run
results/cpu/           the CPU results above
results/gpu/           the GPU results: summary.md, results.json, every answer in answers.jsonl, and the raw runs
```

## Sources

- [T²-RAGBench: Text-and-Table Benchmark for Evaluating RAG](https://arxiv.org/abs/2506.12071)
- [From BM25 to Corrective RAG: Benchmarking Retrieval Strategies for Text-and-Table Documents](https://arxiv.org/pdf/2604.01733)
- [TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning](https://aclanthology.org/2025.emnlp-main.710.pdf)
- [Semantically Similar, Logically Distinct: Diagnosing the Semantic-Answerability Gap in Table RAG](https://arxiv.org/pdf/2607.17742)
- [Structure-Aware Chunking for Tabular Data in RAG](https://arxiv.org/html/2605.00318v1)
- [Docling](https://github.com/docling-project/docling)
