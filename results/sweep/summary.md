> Copied from the run's console output. The raw files of this run (results.json, answers.jsonl) were lost when
> the Colab runtime was reset before the download, so only this summary exists.

# Retrieval sweep

Tuned on the dev half of the documents, reported on the test half. `@N`: evidence inside the first N characters.
BM25 weight: 0 = embeddings only, 1 = keywords only.

## Dev half: every setting

| Embedding model | BM25 weight | Reranker | specqa @1000 | specqa @2000 | t2 @1000 | t2 @2000 | specqa MRR | t2 MRR |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BAAI/bge-small-en-v1.5` | 0.0 | - | 79.4% | 95.5% | 97.3% | 99.2% | 0.773 | 0.948 |
| `BAAI/bge-small-en-v1.5` | 0.3 | - | 80.6% | 95.5% | 98.4% | 98.9% | 0.777 | 0.975 |
| `BAAI/bge-small-en-v1.5` | 0.5 | - | 80.6% | 94.2% | 98.4% | 99.2% | 0.753 | 0.976 |
| `BAAI/bge-small-en-v1.5` | 0.7 | - | 81.3% | 92.3% | 98.1% | 99.5% | 0.693 | 0.977 |
| `BAAI/bge-small-en-v1.5` | 1.0 | - | 80.0% | 91.6% | 96.8% | 98.6% | 0.644 | 0.962 |
| `BAAI/bge-small-en-v1.5` | 0.7 | yes | 79.4% | 92.9% | 97.0% | 98.6% | 0.762 | 0.970 |
| `BAAI/bge-base-en-v1.5` | 0.0 | - | 82.6% | 94.8% | 97.8% | 99.2% | 0.624 | 0.965 |
| `BAAI/bge-base-en-v1.5` | 0.3 | - | 87.1% | 95.5% | 98.4% | 99.2% | 0.706 | 0.973 |
| `BAAI/bge-base-en-v1.5` | 0.5 | - | 86.5% | 94.8% | 98.1% | 98.9% | 0.719 | 0.967 |
| `BAAI/bge-base-en-v1.5` | 0.7 | - | 81.9% | 94.2% | 98.4% | 98.9% | 0.698 | 0.967 |
| `BAAI/bge-base-en-v1.5` | 1.0 | - | 80.0% | 91.6% | 96.8% | 98.6% | 0.644 | 0.962 |
| `BAAI/bge-base-en-v1.5` | 0.3 | yes | 79.4% | 92.9% | 97.0% | 98.6% | 0.762 | 0.969 |
| `intfloat/e5-base-v2` | 0.0 | - | 87.1% | 96.1% | 97.0% | 99.5% | 0.796 | 0.972 |
| `intfloat/e5-base-v2` | 0.3 | - | 87.1% | 95.5% | 99.5% | 99.7% | 0.787 | 0.985 |
| `intfloat/e5-base-v2` | 0.5 | - | 85.8% | 96.1% | 99.5% | 99.7% | 0.755 | 0.985 |
| `intfloat/e5-base-v2` | 0.7 | - | 85.2% | 94.8% | 98.6% | 99.5% | 0.712 | 0.979 |
| `intfloat/e5-base-v2` | 1.0 | - | 80.0% | 91.6% | 96.8% | 98.6% | 0.644 | 0.962 |
| `intfloat/e5-base-v2` | 0.3 | yes | 79.4% | 92.9% | 96.8% | 98.4% | 0.762 | 0.968 |
| `BAAI/bge-m3` | 0.0 | - | 93.5% | 99.4% | 96.8% | 99.5% | 0.874 | 0.966 |
| `BAAI/bge-m3` | 0.3 | - | 89.0% | 98.1% | 98.1% | 99.7% | 0.843 | 0.978 |
| `BAAI/bge-m3` | 0.5 | - | 87.1% | 97.4% | 98.6% | 99.5% | 0.785 | 0.978 |
| `BAAI/bge-m3` | 0.7 | - | 87.1% | 96.8% | 98.6% | 99.5% | 0.749 | 0.976 |
| `BAAI/bge-m3` | 1.0 | - | 80.0% | 91.6% | 96.8% | 98.6% | 0.644 | 0.962 |
| `BAAI/bge-m3` | 0.0 | yes | 79.4% | 92.9% | 97.0% | 98.4% | 0.762 | 0.969 |
| `intfloat/multilingual-e5-base` | 0.0 | - | 83.2% | 95.5% | 96.5% | 98.9% | 0.740 | 0.962 |
| `intfloat/multilingual-e5-base` | 0.3 | - | 85.8% | 95.5% | 98.4% | 99.2% | 0.808 | 0.984 |
| `intfloat/multilingual-e5-base` | 0.5 | - | 84.5% | 96.1% | 98.4% | 99.2% | 0.773 | 0.981 |
| `intfloat/multilingual-e5-base` | 0.7 | - | 83.9% | 96.1% | 98.1% | 99.5% | 0.703 | 0.979 |
| `intfloat/multilingual-e5-base` | 1.0 | - | 80.0% | 91.6% | 96.8% | 98.6% | 0.644 | 0.962 |
| `intfloat/multilingual-e5-base` | 0.3 | yes | 79.4% | 92.9% | 96.8% | 98.9% | 0.762 | 0.968 |

Chosen: `BAAI/bge-m3`, BM25 weight 0.0, reranker off (highest mean found@1000 over the suites).

## Test half: chosen against today's defaults

| Setting | Embedding model | BM25 weight | Reranker | specqa @1000 | specqa @2000 | t2 @1000 | t2 @2000 | specqa MRR | t2 MRR |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| defaults | `BAAI/bge-small-en-v1.5` | 0.5 | - | 84.6% | 90.7% | 97.5% | 99.7% | 0.712 | 0.974 |
| chosen | `BAAI/bge-m3` | 0.0 | - | 89.5% | 95.7% | 98.1% | 99.2% | 0.840 | 0.978 |
