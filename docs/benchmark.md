# Benchmark

How the pipeline is compared with other approaches, the full results, and what they do and don't show.
Raw numbers: `results/cpu/summary.md` and `results/gpu/summary.md`.

## Setup

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

What surprised me most: **row context matters more than anything else.** I expected stitching split tables to be the big win. Without the section name on each row, my
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
(see "What's next").

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
- **On SpecQA the check adds little.** Without it, slightly more answers are right (96.4%) and slightly more are
  wrong (3.6%). My yes/no rule is too strict: correct yes/no answers drop from 81% to 67%.

How the GPU numbers were made: one full run with every method, then a second run of only my method after
the fix to the check (`results/gpu/runs/`). `python -m tablerag.bench.merge` combines them and rescores every
method with the same rules, so a reply like "NOT_FOUND" counts as a refusal for all of them.

## Limits of this comparison

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

- **Measure the new steps on the GPU.** Checkable calculations, filtered counts, totals, means and the query
  clean-up are built and tested, but the GPU results above are from before them. The calculation step targets
  exactly the FinQA questions that are refused today. The rerun reports it as "ours" against "ours without
  checkable calculations".
- **A risk-coverage curve** instead of one cut-off: precision when answering 50%, 80% or all questions.
- **Docling's table model as the front end** for borderless and scanned tables, with my structure layer on top.
- **Exact operations in SQL.** The rows are already stored with parsed numbers and ranges (see
  [design.md](design.md#storage)); the next step is running counts, maximums and range lookups as SQL queries,
  with a test that they match the Python version.
- **Formulas, symbols and nested tables**, which the current front end doesn't handle.
- **A held-out benchmark on real documents** from different fields, to see whether it generalises.

## Sources

- [T²-RAGBench: Text-and-Table Benchmark for Evaluating RAG](https://arxiv.org/abs/2506.12071)
- [From BM25 to Corrective RAG: Benchmarking Retrieval Strategies for Text-and-Table Documents](https://arxiv.org/pdf/2604.01733)
- [TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning](https://aclanthology.org/2025.emnlp-main.710.pdf)
- [Semantically Similar, Logically Distinct: Diagnosing the Semantic-Answerability Gap in Table RAG](https://arxiv.org/pdf/2607.17742)
- [Structure-Aware Chunking for Tabular Data in RAG](https://arxiv.org/html/2605.00318v1)
- [Docling](https://github.com/docling-project/docling)
