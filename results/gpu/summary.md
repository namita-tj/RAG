# Table RAG benchmark

Retriever: hybrid, reader: Qwen/Qwen2.5-3B-Instruct, the reader sees 2000 characters of retrieved text.

## SpecQA (synthetic datasheets, exact answers)

**Ingestion: does each table row survive in one chunk?**

| Method | Rows intact | Rows intact + section named | Chunks | Mean chunk chars | Sec/page |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed-size text chunks (LangChain style) | 100.0% | 66.9% | 89 | 721.7 | 0.0281 |
| Per-page markdown tables (pdfplumber) | 100.0% | 0.0% | 532 | 121.0 | 0.0291 |
| Camelot tables + text chunks | 100.0% | 0.0% | 391 | 155.4 | 0.4254 |
| Docling + HybridChunker | 98.7% | 82.0% | 224 | 393.6 | 0.726 |
| Ours: structure-aware rows + exact ops + verification | 100.0% | 100.0% | 795 | 267.8 | 0.0277 |
| Ours without row context | 100.0% | 0.0% | 795 | 198.9 | 0.0265 |
| Ours without cross-page stitching | 98.7% | 98.7% | 878 | 260.0 | 0.0312 |
| Ours without table and profile chunks | 100.0% | 100.0% | 619 | 233.7 | 0.0308 |

**Retrieval: is the chunk holding the evidence row within the first N characters retrieved?**

| Method | 1000 chars | 2000 chars | 4000 chars | Median chars needed | 2000 chars, cross-page rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed-size text chunks (LangChain style) | 63.7% | 70.3% | 77.3% | 800 | 25.4% |
| Per-page markdown tables (pdfplumber) | 12.6% | 28.4% | 89.6% | 2677 | 42.9% |
| Camelot tables + text chunks | 9.8% | 26.8% | 88.3% | 2700 | 41.3% |
| Docling + HybridChunker | 65.9% | 81.7% | 82.3% | 735 | 57.1% |
| Ours: structure-aware rows + exact ops + verification | 81.4% | 92.1% | 100.0% | 266 | 94.4% |
| Ours without row context | 0.0% | 26.2% | 72.2% | 2529 | 38.1% |
| Ours without cross-page stitching | 76.3% | 92.4% | 98.1% | 278 | 87.3% |
| Ours without table and profile chunks | 80.4% | 92.1% | 100.0% | 267 | 94.4% |

**Answers**

| Method | Correct | Wrong | Refused | Precision when answered |
| --- | ---: | ---: | ---: | ---: |
| Fixed-size text chunks (LangChain style) | 32.1% | 60.2% | 7.6% | 34.8% |
| Per-page markdown tables (pdfplumber) | 16.0% | 73.0% | 11.0% | 17.9% |
| Camelot tables + text chunks | 15.5% | 75.7% | 8.8% | 17.0% |
| Docling + HybridChunker | 37.1% | 52.8% | 10.1% | 41.2% |
| Ours: ingestion only (plain reader) | 63.1% | 35.3% | 1.6% | 64.2% |
| Ours without verification | 96.4% | 3.6% | 0.0% | 96.4% |
| Ours with the reader citing row ids (earlier design) | 43.6% | 1.3% | 55.1% | 97.0% |
| Ours: structure-aware rows + exact ops + verification | 94.8% | 3.1% | 2.0% | 96.8% |

Correct answers by question type:

| Method | count | covers | flag | max | range | run_in | value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed-size text chunks (LangChain style) | 10.4% | 2.1% | 64.6% | 2.1% | 37.0% | 62.5% | 58.7% |
| Per-page markdown tables (pdfplumber) | 12.5% | 2.1% | 75.0% | 0.0% | 4.9% | 31.2% | 14.1% |
| Camelot tables + text chunks | 10.4% | 0.0% | 83.3% | 0.0% | 3.7% | 31.2% | 12.0% |
| Docling + HybridChunker | 22.9% | 1.0% | 56.2% | 6.2% | 55.6% | 25.0% | 76.1% |
| Ours: ingestion only (plain reader) | 29.2% | 7.3% | 81.2% | 47.9% | 91.4% | 100.0% | 100.0% |
| Ours without verification | 100.0% | 100.0% | 81.2% | 100.0% | 91.4% | 100.0% | 100.0% |
| Ours with the reader citing row ids (earlier design) | 100.0% | 100.0% | 2.1% | 100.0% | 1.2% | 0.0% | 0.0% |
| Ours: structure-aware rows + exact ops + verification | 100.0% | 100.0% | 66.7% | 100.0% | 91.4% | 100.0% | 100.0% |


## T²-RAGBench (FinQA + TAT-DQA, rendered to PDF)

**Retrieval: is a chunk of the right document within the first N characters retrieved?**

| Method | 1000 chars | 2000 chars | 4000 chars | Median chars needed |
| --- | ---: | ---: | ---: | ---: |
| Fixed-size text chunks (LangChain style) | 90.5% | 96.8% | 98.6% | 200 |
| Per-page markdown tables (pdfplumber) | 91.8% | 96.2% | 98.6% | 200 |
| Camelot tables + text chunks | 88.6% | 95.9% | 98.9% | 200 |
| Docling + HybridChunker | 96.3% | 98.8% | 99.5% | 200 |
| Ours: structure-aware rows + exact ops + verification | 97.8% | 99.6% | 99.9% | 200 |
| Ours without row context | 89.3% | 95.3% | 98.4% | 200 |
| Ours without cross-page stitching | 97.8% | 99.6% | 99.9% | 200 |
| Ours without table and profile chunks | 98.1% | 99.5% | 99.9% | 200 |

**Answers**

| Method | Correct | Wrong | Refused | Precision when answered |
| --- | ---: | ---: | ---: | ---: |
| Fixed-size text chunks (LangChain style) | 19.6% | 65.5% | 14.9% | 23.0% |
| Per-page markdown tables (pdfplumber) | 17.9% | 56.2% | 25.9% | 24.2% |
| Camelot tables + text chunks | 16.6% | 52.3% | 31.1% | 24.1% |
| Docling + HybridChunker | 21.5% | 61.9% | 16.6% | 25.8% |
| Ours: ingestion only (plain reader) | 22.1% | 66.6% | 11.4% | 24.9% |
| Ours without verification | 21.6% | 67.0% | 11.4% | 24.4% |
| Ours with the reader citing row ids (earlier design) | 0.1% | 1.8% | 98.1% | 7.1% |
| Ours: structure-aware rows + exact ops + verification | 14.5% | 18.5% | 67.0% | 44.0% |

Correct answers by question type:

| Method | FinQA | TAT-DQA |
| --- | ---: | ---: |
| Fixed-size text chunks (LangChain style) | 6.6% | 28.7% |
| Per-page markdown tables (pdfplumber) | 6.0% | 26.4% |
| Camelot tables + text chunks | 6.6% | 23.6% |
| Docling + HybridChunker | 8.9% | 30.4% |
| Ours: ingestion only (plain reader) | 6.0% | 33.4% |
| Ours without verification | 5.3% | 33.2% |
| Ours with the reader citing row ids (earlier design) | 0.0% | 0.2% |
| Ours: structure-aware rows + exact ops + verification | 1.0% | 24.1% |

