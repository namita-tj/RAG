# Table RAG benchmark

Retriever: bm25, reader: none (retrieval only), the reader sees 2000 characters of retrieved text.

## SpecQA (synthetic datasheets, exact answers)

**Ingestion: does each table row survive in one chunk?**

| Method | Rows intact | Rows intact + section named | Chunks | Mean chunk chars | Sec/page |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed-size text chunks (LangChain style) | 100.0% | 66.9% | 89 | 721.7 | 0.0131 |
| Per-page markdown tables (pdfplumber) | 100.0% | 0.0% | 532 | 121.0 | 0.0185 |
| Camelot tables + text chunks | 100.0% | 0.0% | 203 | 271.9 | 0.2849 |
| Ours: structure-aware rows + exact ops + verification | 100.0% | 100.0% | 795 | 267.8 | 0.0208 |
| Ours without row context | 100.0% | 0.0% | 795 | 198.9 | 0.0226 |
| Ours without cross-page stitching | 98.7% | 98.7% | 878 | 260.0 | 0.0209 |
| Ours without table and profile chunks | 100.0% | 100.0% | 619 | 233.7 | 0.0214 |

**Retrieval: is the chunk holding the evidence row within the first N characters retrieved?**

| Method | 1000 chars | 2000 chars | 4000 chars | Median chars needed | 2000 chars, cross-page rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed-size text chunks (LangChain style) | 65.3% | 69.7% | 74.8% | 795 | 23.8% |
| Per-page markdown tables (pdfplumber) | 18.6% | 49.8% | 95.6% | 2019 | 66.7% |
| Camelot tables + text chunks | 3.2% | 18.6% | 83.6% | 2893 | 23.8% |
| Ours: structure-aware rows + exact ops + verification | 77.3% | 92.1% | 100.0% | 417 | 88.1% |
| Ours without row context | 0.3% | 13.6% | 58.0% | 3311 | 19.8% |
| Ours without cross-page stitching | 71.0% | 89.3% | 98.7% | 457 | 88.1% |
| Ours without table and profile chunks | 75.7% | 92.7% | 100.0% | 410 | 88.1% |

Exact structured path (no language model): took 192 of 445 questions, 192 correct (100.0%).

