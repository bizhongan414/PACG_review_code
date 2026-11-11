"""Verifiable answer reward, with optional explicitly configured format reward."""
import re

from mathruler.grader import extract_boxed_content, grade_answer


def compute_score(reward_inputs, format_weight=0.0):
    if not isinstance(reward_inputs, list) or not 0 <= format_weight <= 1:
        raise ValueError("Use batch rewards and format_weight in [0,1]")
    scores = []
    for row in reward_inputs:
        response = row["response"]
        accuracy = float(grade_answer(extract_boxed_content(response), row["ground_truth"]))
        fmt = float(re.fullmatch(r"\s*<think>[\s\S]+?</think>\s*\\boxed\{[\s\S]+\}\s*", response) is not None)
        scores.append({"overall": (1 - format_weight) * accuracy + format_weight * fmt,
                       "accuracy": accuracy, "format": fmt})
    return scores

