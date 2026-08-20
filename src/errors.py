"""Per-sample failure recording.

A baseline run must not stop for one bad sample, and it must never paper over
one either: a backbone that returns unparsable JSON produces no artifact at all,
and the reason is written as ``error.log`` in the index directory of every
modality it cost us. A T2I crash only costs the image, so only ``image/<index>/``
gets the log; a backbone failure costs all three. The evaluation harness resolves
candidates by extension, so a ``.log`` is invisible to it -- an affected modality
simply reads as missing, which is what it is.
"""

import json
import os
import traceback
from datetime import datetime
from typing import Any, Dict, Iterable, List, Sequence

MODALITIES = ("text", "image", "speech")

ERROR_LOG_NAME = "error.log"
ERROR_INDEX_NAME = "errors.jsonl"


class BackboneOutputError(RuntimeError):
    """The backbone returned output no artifact can be built from.

    Raised instead of substituting placeholder text, prompts or speech.
    """

    def __init__(self, message: str, raw_output: Any = None):
        super().__init__(message)
        self.raw_output = raw_output


def set_modalities(exc: BaseException, modalities: Iterable[str]) -> BaseException:
    """Record which modalities an in-flight failure costs us.

    Untagged failures are treated as costing all three, so a stage that cannot
    carry the tag degrades to the safe answer rather than under-reporting.
    """
    try:
        exc.modalities = tuple(modalities)
    except Exception:
        pass
    return exc


def affected_modalities(exc: BaseException) -> Sequence[str]:
    tagged = getattr(exc, "modalities", None)
    if not tagged:
        return MODALITIES
    return tuple(m for m in MODALITIES if m in tagged)


def write_error_log(output_dir_base: str, entry: Dict[str, Any], exc: BaseException, config_name: str = "") -> List[str]:
    """Record a sample's failure and return the paths of the logs written.

    Args:
        output_dir_base: This baseline's results directory.
        entry: The sample being processed.
        exc: The exception that ended it.
        config_name: Config the run was launched with.
    """
    index = entry["index"]
    modalities = affected_modalities(exc)

    raw_output = getattr(exc, "raw_output", None)
    lines = [
        f"time      : {datetime.now().isoformat(timespec='seconds')}",
        f"config    : {config_name}",
        f"index     : {index}",
        f"sample    : {entry.get('id', '')}",
        f"affected  : {', '.join(modalities)}",
        f"error     : {type(exc).__name__}: {exc}",
        "",
        "traceback:",
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip(),
    ]
    if raw_output is not None:
        lines += ["", "raw backbone output:", str(raw_output)]
    report = "\n".join(lines) + "\n"

    log_paths = []
    for modality in modalities:
        log_dir = os.path.join(output_dir_base, modality, str(index))
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, ERROR_LOG_NAME)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(report)
        log_paths.append(log_path)

    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "index": index,
        "sample": entry.get("id", ""),
        "modalities": list(modalities),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "logs": log_paths,
    }
    with open(os.path.join(output_dir_base, ERROR_INDEX_NAME), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return log_paths
