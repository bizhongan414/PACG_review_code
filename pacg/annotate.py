"""Annotate fixed responses through a configurable OpenAI-compatible multimodal API."""
import argparse
import json
import os
import time
from pathlib import Path

import requests

from .annotation import build_prompt, image_to_data_url, parse_json_content


def align_exact(trace, spans):
    """Resolve exact text; honor supplied offsets and reject ambiguous repetitions."""
    result = []
    for raw in spans:
        span = dict(raw)
        text = span.get("text", "")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty or invalid annotation span")
        start = span.get("char_start")
        if start is None:
            if trace.count(text) != 1:
                raise ValueError("Span is missing or ambiguous; supply verified char_start/char_end")
            start = trace.index(text)
        if not isinstance(start, int):
            raise ValueError("char_start must be an integer")
        end = span.get("char_end", start + len(text))
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or trace[start:end] != text:
            raise ValueError("Annotation text/offset mismatch")
        span.update(char_start=start, char_end=end)
        result.append(span)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--url", default=os.getenv("ANNOTATOR_URL", "http://127.0.0.1:8000/v1/chat/completions"))
    parser.add_argument("--model", default=os.getenv("ANNOTATOR_MODEL", "annotator"))
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if args.retries < 0 or args.timeout <= 0:
        parser.error("retries must be nonnegative and timeout positive")
    session = requests.Session()
    session.trust_env = False
    key = os.getenv("ANNOTATOR_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    root = Path(args.input).resolve().parent
    with open(args.input) as source, open(args.output, "x") as dest:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            image = Path(row["image"])
            if not image.is_absolute():
                image = root / image
            payload = {"model": args.model, "temperature": args.temperature,
                       "chat_template_kwargs": {"enable_thinking": False}, "messages": [
                           {"role": "user", "content": [
                               {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
                               {"type": "text", "text": build_prompt(row, row["response"], len(row["response"]))}]}]}
            for attempt in range(args.retries + 1):
                try:
                    response = session.post(args.url, headers=headers, json=payload, timeout=(10, args.timeout))
                    if not response.ok:
                        raise RuntimeError(f"Annotator HTTP {response.status_code}")
                    content = response.json()["choices"][0]["message"]["content"]
                    parsed = parse_json_content(content)
                    if not isinstance(parsed.get("spans"), list):
                        raise ValueError("Annotation has no spans list")
                    row["spans"] = align_exact(row["response"], parsed["spans"])
                    break
                except (requests.RequestException, ValueError, KeyError, IndexError, RuntimeError):
                    if attempt == args.retries:
                        raise
                    time.sleep(min(2 ** attempt, 8))
            row["image"] = str(image.resolve())
            row["annotation"] = {"model": args.model, "temperature": args.temperature}
            dest.write(json.dumps(row, ensure_ascii=False) + "\n")
            dest.flush()


if __name__ == "__main__":
    main()

