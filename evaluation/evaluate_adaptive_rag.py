from src.adaptive_rag import adaptive_qa_pipeline

test_queries = [
    {"question": "What is the emission limit for 150 kHz?", "answer": "66 dBμV"},
    {"question": "Who signed the contract?", "answer": "Dr. Müller"},
    # Add more examples
]

correct = 0
for i, ex in enumerate(test_queries):
    result = adaptive_qa_pipeline(ex["question"])
    pred = result["answer"].strip().lower()
    gold = ex["answer"].strip().lower()

    print(f"[{i+1}] Q: {ex['question']}")
    print(f"Predicted: {pred}")
    print(f"Expected:  {gold}")
    print(f"Retrieved? {result['retrieved']} | Entropy: {result['entropy']:.2f}")
    print("-" * 40)

    if pred == gold:
        correct += 1

print(f"Accuracy: {correct}/{len(test_queries)}")
