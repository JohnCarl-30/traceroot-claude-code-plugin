"""
Stop can fire before Claude Code has written the turn's final assistant rows to
the transcript. handle_stop must wait for them (the payload's
last_assistant_message says what the last row should contain) instead of
emitting a truncated trace and marking the ask as done.
"""
import json

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from traceroot_observability import emit, spans, state

RECORDS = [
    {"type": "user", "timestamp": "2026-06-22T00:00:00Z",
     "message": {"role": "user", "content": "read the file"}},
    {"type": "assistant", "requestId": "r1", "timestamp": "2026-06-22T00:00:01Z",
     "message": {"id": "r1", "model": "claude-opus-4-8",
                 "content": [{"type": "text", "text": "Reading it."},
                             {"type": "tool_use", "id": "tu-1", "name": "Read", "input": {}}],
                 "usage": {"input_tokens": 5, "output_tokens": 3}}},
    {"type": "user", "timestamp": "2026-06-22T00:00:02Z",
     "message": {"role": "user",
                 "content": [{"type": "tool_result", "tool_use_id": "tu-1", "content": "data"}]}},
    {"type": "assistant", "requestId": "r2", "timestamp": "2026-06-22T00:00:03Z",
     "message": {"id": "r2", "model": "claude-opus-4-8",
                 "content": [{"type": "text", "text": "final answer"}],
                 "usage": {"input_tokens": 5, "output_tokens": 3}}},
]


def _setup(tmp_path, monkeypatch, reads):
    """Wire handle_stop to an in-memory exporter; `reads` is what each successive
    transcript read returns (the last entry repeats)."""
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)
    monkeypatch.setenv("TRACEROOT_API_KEY", "test-key")
    exp = InMemorySpanExporter()

    def fake_tracer(cfg, git):
        prov = TracerProvider()
        prov.add_span_processor(SimpleSpanProcessor(exp))
        return prov.get_tracer("t"), (lambda: None)

    calls = []

    def fake_read(path):
        calls.append(path)
        return reads[min(len(calls), len(reads)) - 1]

    monkeypatch.setattr(emit, "_tracer", fake_tracer)
    monkeypatch.setattr(emit, "_resolve_git", lambda payload, st: ("", ""))
    monkeypatch.setattr(emit._t, "read_all_records", fake_read)
    monkeypatch.setattr(emit.time, "sleep", lambda s: None)
    return exp, calls


def _root_output(exp):
    roots = [s for s in exp.get_finished_spans() if s.attributes.get(spans.OI["KIND"]) == "AGENT"]
    assert len(roots) == 1
    return roots[0].attributes.get(spans.OI["OUTPUT"])


def test_stop_waits_for_final_assistant_rows(tmp_path, monkeypatch):
    # First two reads lack the final assistant row; the third has it.
    exp, calls = _setup(tmp_path, monkeypatch, [RECORDS[:3], RECORDS[:3], RECORDS])

    emit.handle_stop({"hook_event_name": "Stop", "session_id": "sess-lag",
                      "transcript_path": str(tmp_path / "t.jsonl"),
                      "last_assistant_message": "final answer"})

    assert len(calls) == 3
    assert _root_output(exp) == "final answer"


def test_stop_reads_once_without_last_assistant_message(tmp_path, monkeypatch):
    # Older Claude Code sends no last_assistant_message: nothing to wait for.
    exp, calls = _setup(tmp_path, monkeypatch, [RECORDS])

    emit.handle_stop({"hook_event_name": "Stop", "session_id": "sess-nolam",
                      "transcript_path": str(tmp_path / "t.jsonl")})

    assert len(calls) == 1
    assert _root_output(exp) == "final answer"
