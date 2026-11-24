"""The shared train/inference contract for a frozen Tau0 planner's text output."""
import json


class SubtaskConditionError(ValueError):
    """Missing or invalid planner conditioning; never retry with another training frame."""


def normalize_subtask(value, *, required=False):
    if value is None:
        if required:
            raise SubtaskConditionError("This model requires a high_level_subtask from Tau0 Proposal")
        return None
    if not isinstance(value, str) or not value.strip():
        raise SubtaskConditionError("high_level_subtask must be a non-empty string")
    value = value.strip()
    if len(value) > 4096:
        raise SubtaskConditionError("high_level_subtask exceeds 4096 characters")
    return value


def condition_code_prompt(prompt, subtask, *, tokenizer=None, max_subtask_tokens=128, required=False):
    """Insert a single escaped comment BEFORE # Next; all of this remains prompt-side.

    JSON escaping prevents a multi-line planner answer from changing the prompt's stage
    delimiters. Keeping the same function in both paths avoids a train/serve mismatch.
    None preserves the original prompt byte-for-byte for baseline checkpoints.
    """
    subtask = normalize_subtask(subtask, required=required)
    if subtask is None:
        return prompt
    if tokenizer is not None:
        try:
            ids = tokenizer.encode(subtask, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(subtask)
        if len(ids) > max_subtask_tokens:
            raise SubtaskConditionError(
                f"Tau0 subtask uses {len(ids)} tokens, limit={max_subtask_tokens}; "
                "shorten the proposal or increase subtask_max_tokens and the prompt budget"
            )
    line = "# High-level subtask (Tau0): " + json.dumps(subtask, ensure_ascii=False) + "\n"
    marker = "# Next:\n"
    if prompt.endswith(marker):
        return prompt[:-len(marker)] + line + marker
    return prompt + ("" if prompt.endswith("\n") else "\n") + line
