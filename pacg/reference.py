"""Shared fixed-reference IO and single-image prompting."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_rows(filename):
    root = Path(filename).resolve().parent
    rows = []
    seen = set()
    with open(filename) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample = str(row["sample_id"])
            if sample in seen:
                raise ValueError(f"Repeated sample_id: {sample}")
            seen.add(sample)
            image = Path(row["image"])
            row["image"] = str(image if image.is_absolute() else root / image)
            if not Path(row["image"]).is_file():
                raise FileNotFoundError(row["image"])
            row.setdefault("question_id", sample)
            rows.append(row)
    if not rows:
        raise ValueError("Empty reference")
    return rows


def messages(question):
    from jinja2 import Template
    template = (ROOT / "examples/format_prompt/math_format_perception.jinja").read_text()
    # Preserve the training dataset's ordering of image and text content.
    if "<image>" not in question:
        question = "<image>\n" + question
    if question.count("<image>") != 1:
        raise ValueError("The standalone fixed-reference diagnostic supports one image per sample")
    formatted = Template(template).render(content=question)
    content = []
    for index, text in enumerate(formatted.split("<image>")):
        if index:
            content.append({"type": "image"})
        if text:
            content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def digest(row):
    record = {key: row.get(key) for key in ("sample_id", "question_id", "benchmark", "question", "response", "spans")}
    record["image_sha256"] = hashlib.sha256(Path(row["image"]).read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

