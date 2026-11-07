# Design

## Why I built this

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

## The pipeline

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
- **Query clean-up** (`src/tablerag/query.py`). Numbers and units are rewritten the way documents print them:
  "1,450 revolutions per minute" becomes "1450 rpm", "5 bars" becomes "5 bar". Only a number followed by a unit
  is touched. An optional cross-encoder reranker re-orders the top 30 candidates, which helps with near-identical
  rows like "Stage 4" and "Stage 14".
- **Exact answers without the language model** (`src/tablerag/structured_ops.py`). Computed from the stored rows,
  with unit conversion:
  - range lookups ("which stage covers 1450 rpm"),
  - counts, also filtered by a value ("how many stages allow intermittent duty") or a threshold ("how many have
    a maximum pressure above 5 bar", also given as "600 kPa"),
  - totals and means of a column,
  - highest and lowest values, and which row has them (all tied rows are listed).

  The kind of question decides the operation: a count question is only ever answered by counting. When the
  question doesn't clearly match one table, one column and one operation, this step does nothing and the normal
  path answers instead. It never guesses.
- **Check every answer against the evidence** (`src/tablerag/answer.py`). After the model answers, the code
  checks that the value is actually printed in a retrieved row or passage. If several rows print it, the row
  the question names wins. A yes/no answer is only kept when the row the question asks about was retrieved.
  Otherwise the system says "NOT FOUND", which you can check, instead of a number that might be wrong.
- **Checkable calculations** (`src/tablerag/calc.py`). A small model is bad at arithmetic, and a calculated
  answer is printed nowhere, so the check above would refuse it. When the model's number isn't in the evidence,
  it is asked for the calculation instead, for example `(5829 - 5735) / 5735 * 100`. The code evaluates it safely
  (only numbers, + - * / and parentheses, nothing that can run as code) and keeps the result only if every number
  in the formula is printed in the evidence.

## Storage

I started from what retrieval and answering need from storage: keyword search with my own tokenizer, exact
vector search, the same chunk ids for both, numeric queries on cells for the exact operations, the link from a
chunk to its rows for the evidence check, and the same answer every time.

- A **vector database** covers only the vector part, and usually searches approximately (HNSW), which can miss
  the right chunk. It can't count rows, find maximums or do unit-aware range lookups, so the tables would need
  a second store anyway.
- A **document database** handles tables with different columns well, but needs a server and has no keyword
  ranking without extra products.
- **Several specialised databases** would each be best at one job, but every document update would have to land
  in all of them, and a failure halfway leaves them out of sync.

So everything lives in one SQLite file (`src/tablerag/store.py`) with three views of the same data:

| View | Tables | Used by |
| --- | --- | --- |
| Tables and cells | `tables`, `columns`, `rows`, `cells` | exact operations, evidence check, sources |
| Chunks | `chunks`, `chunk_rows` | keyword search, evidence check |
| Embeddings | `embeddings` (per chunk and model) | semantic search, brute force and exact |

- Every cell keeps its text exactly as printed. Next to it sit the parsed number or range, also converted to the
  base unit of its quantity, so "which row's speed range contains 1450 rpm" is one SQL query.
- I deliberately don't use SQLite's built-in full-text ranking: its tokenizer splits "1.5" into "1" and "5", and
  its BM25 parameters differ from mine, so rankings would change. Instead the index loads the same objects
  ingestion produces, and my BM25 and the dense search are rebuilt from them.
- On all 445 SpecQA questions, the stored index gives identical retrieval and identical exact answers to the
  in-memory version, and `tests/test_store.py` checks this on every push.
- A changed document is replaced in one transaction. An index built by an older pipeline version is refused
  instead of mixed with new data.
- If this ever needs to serve many users, the same design moves to PostgreSQL with pgvector, still one database.

## Looking back

If I started again, I would begin with the evaluation, not the pipeline. First the scope: what data I have, what
I benchmark against, how I measure performance and how I check whether it generalises across industries and
document types, and whether it scales. Then build. I would also test much earlier which parts need a
language model at all. In this project the deterministic parts (the exact table operations and the check done
in code) moved the accuracy most, and the one time I relied on the model to do bookkeeping, citing row ids,
it failed.
