> Copied from the run's console output. The raw files of this run (results.json, answers.jsonl) were lost when
> the Colab runtime was reset before the download, so only this summary exists.

# Table RAG benchmark

Retriever: hybrid, reader: Qwen/Qwen2.5-3B-Instruct, the reader sees 2000 characters of retrieved text.

## SpecQA (synthetic datasheets, exact answers)

**Ingestion: does each table row survive in one chunk?**

| Method | Rows intact | Rows intact + section named | Chunks | Mean chunk chars | Sec/page |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ours: structure-aware rows + exact ops + verification | 100.0% | 100.0% | 795 | 267.8 | 0.0332 |

**Retrieval: is the chunk holding the evidence row within the first N characters retrieved?**

| Method | 1000 chars | 2000 chars | 4000 chars | Median chars needed | 2000 chars, cross-page rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ours: structure-aware rows + exact ops + verification | 81.4% | 92.1% | 100.0% | 266 | 94.4% |

**Answers**

| Method | Correct | Wrong | Refused | Precision when answered |
| --- | ---: | ---: | ---: | ---: |
| Ours without checkable calculations | 94.8% | 3.1% | 2.0% | 96.8% |
| Ours without verification | 96.4% | 3.6% | 0.0% | 96.4% |
| Ours: ingestion only (plain reader) | 63.1% | 35.3% | 1.6% | 64.2% |
| Ours: structure-aware rows + exact ops + verification | 94.8% | 3.4% | 1.8% | 96.6% |

Correct answers by question type:

| Method | count | covers | flag | max | range | run_in | value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours without checkable calculations | 100.0% | 100.0% | 66.7% | 100.0% | 91.4% | 100.0% | 100.0% |
| Ours without verification | 100.0% | 100.0% | 81.2% | 100.0% | 91.4% | 100.0% | 100.0% |
| Ours: ingestion only (plain reader) | 29.2% | 7.3% | 81.2% | 47.9% | 91.4% | 100.0% | 100.0% |
| Ours: structure-aware rows + exact ops + verification | 100.0% | 100.0% | 66.7% | 100.0% | 91.4% | 100.0% | 100.0% |


## T²-RAGBench (FinQA + TAT-DQA, rendered to PDF)

**Retrieval: is a chunk of the right document within the first N characters retrieved?**

| Method | 1000 chars | 2000 chars | 4000 chars | Median chars needed |
| --- | ---: | ---: | ---: | ---: |
| Ours: structure-aware rows + exact ops + verification | 97.8% | 99.6% | 99.9% | 200 |

**Answers**

| Method | Correct | Wrong | Refused | Precision when answered |
| --- | ---: | ---: | ---: | ---: |
| Ours without checkable calculations | 13.8% | 22.9% | 63.3% | 37.7% |
| Ours without verification | 20.1% | 69.2% | 10.7% | 22.5% |
| Ours: ingestion only (plain reader) | 21.9% | 66.7% | 11.4% | 24.7% |
| Ours: structure-aware rows + exact ops + verification | 29.9% | 44.8% | 25.3% | 40.0% |

Correct answers by question type:

| Method | FinQA | TAT-DQA |
| --- | ---: | ---: |
| Ours without checkable calculations | 1.0% | 22.9% |
| Ours without verification | 5.0% | 30.8% |
| Ours: ingestion only (plain reader) | 6.0% | 33.2% |
| Ours: structure-aware rows + exact ops + verification | 18.2% | 38.1% |
