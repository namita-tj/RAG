"""Turning retrieved chunks into an answer, and (our method) refusing answers that cannot be traced.

Every method uses the same reader model and the same prompt. Our method adds two things:

* questions the stored tables can answer exactly (range lookups, counts, extremes) are
  answered by structured_ops without the model;
* every reader answer is checked against the retrieved evidence: it is only kept if the
  value is printed in a retrieved table row or passage ("verified"). Otherwise the system
  says it could not find the answer. In a specification, "not found" can be checked; a
  wrong number might not be.

An earlier version asked the model to cite the row id itself (mode "ours_cite", kept as an
ablation). Small readers then refused most questions they could answer, so the check is now
done in code and the model only has to answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from tablerag.ingest.structured import derived_fields
from tablerag.normalize import clean, norm_key, numbers_in, same_value
from tablerag.retrieve import Retriever
from tablerag.structured_ops import try_structured
from tablerag.types import Chunk, StoredRow, StoredTable

NOT_FOUND = "NOT FOUND"

PROMPT = """Answer the question using only the context below. Reply with only the answer: a number, a name or yes/no, without explanation. If a calculation is needed, reply with only the final number. If the context does not contain the answer, reply {nf}.

Context:
{context}

Question: {question}
Answer:"""

CITE_PROMPT = """Answer the question using only the context below. Table rows start with an id in square brackets. Reply in the form: answer | row id. Reply with only the answer value (a number, a name or yes/no) and the id of the row it comes from, without explanation. If a calculation is needed, reply with only the final number and the id of one row you used. If the context does not contain the answer, reply {nf}.

Context:
{context}

Question: {question}
Answer:"""


class Reader(Protocol):
    def generate(self, prompts: list[str]) -> list[str]: ...


@dataclass
class Answer:
    text: str
    route: str  # "reader", "structured", "verified", "refused", "unverified"
    evidence: list[Chunk] = field(default_factory=list)
    rows: list[str] = field(default_factory=list)  # row ids the answer is traced to
    raw: str = ""


def within_budget(chunks: list[Chunk], limit: int) -> list[Chunk]:
    """The chunks (in order) whose text starts inside the first ``limit`` characters."""
    out, used = [], 0
    for c in chunks:
        if used >= limit:
            break
        out.append(c)
        used += len(c.text.strip())
    return out


def build_context(chunks: list[Chunk], limit: int = 2000) -> str:
    parts, used = [], 0
    for i, c in enumerate(chunks, 1):
        text = c.text.strip()
        if used + len(text) > limit:
            text = text[: max(0, limit - used)]
        parts.append(f"({i}) {text}")
        used += len(text)
        if used >= limit:
            break
    return "\n\n".join(parts)


def parse_reply(reply: str) -> tuple[str, str | None]:
    """'100 | doc:t1:r3' -> ('100', 'doc:t1:r3'); also accepts '[doc:t1:r3]'."""
    line = clean(reply.strip().splitlines()[0] if reply.strip() else "")
    line = line.strip().strip("`").strip()
    m = re.search(r"\[?([\w.\-]+:t\d+:r\d+)\]?", line)
    row = m.group(1) if m else None
    answer = re.split(r"\s*\|\s*", line)[0] if "|" in line else (line.replace(m.group(0), "") if m else line)
    answer = answer.strip().strip(".").strip()
    answer = re.sub(r"^(answer|the answer is)\s*[:\-]?\s*", "", answer, flags=re.IGNORECASE)
    return answer, row


def is_not_found(answer: str) -> bool:
    a = norm_key(answer).replace("_", " ")
    return not a or "not found" in a or a in {"unknown", "n/a", "none"}


def _row_supports(row: StoredRow, answer: str) -> bool:
    if norm_key(answer) in {"yes", "no"}:
        return True  # yes/no comes from the cited row's flags; the citation itself is checked
    values = list(row.cells.values())
    for h, v in row.cells.items():
        values += list(derived_fields(h, v).values())
    if any(same_value(answer, v) for v in values):
        return True
    nums = numbers_in(answer)
    cell_nums = [n for v in values for n in numbers_in(v)]
    # "900 - 1500 rpm" is supported when every number in it is printed in the row
    return bool(nums) and all(any(n == c for c in cell_nums) for n in nums)


def verify(answer: str, cited: str | None, evidence: list[Chunk], rows: dict[str, StoredRow]) -> str | None:
    """The row id that supports the answer, or None. The cited row is tried first; if the
    citation is wrong but exactly one retrieved row contains the answer, that row is used."""
    retrieved = [rid for c in evidence for rid in c.row_ids]
    if cited and cited in rows and cited in retrieved and _row_supports(rows[cited], answer):
        return cited
    if norm_key(answer) in {"yes", "no"}:
        return None
    support = [rid for rid in dict.fromkeys(retrieved) if rid in rows and _row_supports(rows[rid], answer)]
    return support[0] if len(support) == 1 else None


def _names_row(question: str, row: StoredRow) -> bool:
    """True if the question names this row ("... of Stage 4 ..."), not just a prefix of it."""
    label = norm_key(row.label)
    return bool(label) and re.search(rf"(?<![\w]){re.escape(label)}(?![\w])", norm_key(question)) is not None


def ground(answer: str, question: str, evidence: list[Chunk], rows: dict[str, StoredRow]) -> str | None:
    """Checks a reader answer against what was retrieved. Returns the supporting row id,
    "text" when the value is printed in a retrieved passage, or None when nothing retrieved
    contains it.

    * a value or range must be printed in a retrieved row; if several rows print it, the one
      the question names is preferred;
    * yes/no cannot be matched against a cell, so it is only kept when exactly one retrieved
      row is the row the question names;
    * otherwise every number in the answer (or the answer text) must appear in a retrieved
      passage. A calculated result that is printed nowhere is refused.
    """
    retrieved = [rows[rid] for rid in dict.fromkeys(rid for c in evidence for rid in c.row_ids) if rid in rows]
    named = [r for r in retrieved if _names_row(question, r)]
    if norm_key(answer) in {"yes", "no"}:
        return named[0].row_id if len(named) == 1 else None
    support = [r for r in retrieved if _row_supports(r, answer)]
    if support:
        preferred = [r for r in support if r in named]
        return (preferred or support)[0].row_id
    text = " ".join(c.text for c in evidence)
    nums = numbers_in(answer)
    if nums:
        printed = numbers_in(text)
        return "text" if all(any(n == p for p in printed) for n in nums) else None
    key = norm_key(answer)
    return "text" if len(key) >= 2 and key in norm_key(text) else None


def answer_questions(
    questions: list[dict],
    retriever: Retriever,
    reader: Reader | None,
    k: int = 5,
    tables: list[StoredTable] | None = None,
    mode: str = "plain",
    batch_size: int = 16,
    scoped: bool = True,
    context_chars: int = 2000,
) -> list[Answer]:
    """Answers questions (each a dict with "question" and "doc_id").

    scoped=True searches only the question's document (SpecQA: the user knows which spec);
    scoped=False searches the whole collection (T²-RAGBench) and the structured path only
    sees tables that appear in the retrieved chunks.

    mode "plain": retrieve + reader (all baselines).
    mode "ours":  structured operations first, then retrieve + reader + check against the evidence.
    mode "ours_unverified": like "ours" but keeps every reader answer (ablation).
    mode "ours_cite": like "ours" but the reader must cite a row id (the earlier design, ablation).
    """
    rows = {r.row_id: r for t in (tables or []) for r in t.rows}
    by_doc: dict[str, list[StoredTable]] = {}
    for t in tables or []:
        by_doc.setdefault(t.doc_id, []).append(t)
    results: list[Answer | None] = [None] * len(questions)
    pending: list[tuple[int, list[Chunk], str]] = []
    for i, q in enumerate(questions):
        hits = [c for c, _ in retriever.search(q["question"], k, q.get("doc_id") if scoped else None)]
        evidence = within_budget(hits, context_chars)  # what the reader will see
        if mode != "plain":
            if scoped:
                candidates = by_doc.get(q.get("doc_id", ""), [])
            else:
                seen = {c.meta.get("table_id") for c in evidence}
                candidates = [t for t in (tables or []) if t.table_id in seen]
            op = try_structured(q["question"], candidates)
            if op is not None:
                results[i] = Answer(op.answer, "structured", evidence, [r.row_id for r in op.rows], op.explanation)
                continue
        template = CITE_PROMPT if mode == "ours_cite" else PROMPT
        pending.append(
            (
                i,
                evidence,
                template.format(nf=NOT_FOUND, context=build_context(evidence, context_chars), question=q["question"]),
            )
        )
    if pending and reader is None:
        raise ValueError("a reader is needed for questions the structured path cannot answer")
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        replies = reader.generate([p for _, _, p in batch]) if reader else []
        for (i, evidence, _), reply in zip(batch, replies, strict=True):
            text, cited = parse_reply(reply)
            if mode == "plain" or is_not_found(text):
                results[i] = Answer(
                    NOT_FOUND if is_not_found(text) else text,
                    "reader" if mode == "plain" else "refused",
                    evidence,
                    raw=reply,
                )
                continue
            if mode == "ours_cite":
                support = verify(text, cited, evidence, rows)
            else:
                support = ground(text, questions[i]["question"], evidence, rows)
            if support:
                results[i] = Answer(text, "verified", evidence, [] if support == "text" else [support], reply)
            elif mode == "ours_unverified":
                results[i] = Answer(text, "unverified", evidence, [], reply)
            else:
                results[i] = Answer(NOT_FOUND, "refused", evidence, [], reply)
    return [r for r in results if r is not None]


class HFReader:
    """A small instruction-tuned model from Hugging Face, greedy decoding (GPU recommended)."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", max_new_tokens: int = 40):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left", truncation_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.torch = torch

    def generate(self, prompts: list[str]) -> list[str]:
        texts = [
            self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=3072).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
        return [self.tok.decode(o[enc["input_ids"].shape[1] :], skip_special_tokens=True).strip() for o in out]
