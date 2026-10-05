"""Unit tests for clean_annotations covering the 4 worked examples."""

from scripts.data.sins.build_manifest import clean_annotations, pick_node


def _approx(segs, expected):
    """Compare list-of-dicts ignoring tiny float diffs in start/end."""
    if len(segs) != len(expected):
        return False, f"len mismatch: got {len(segs)}, expected {len(expected)}"
    for i, (g, e) in enumerate(zip(segs, expected)):
        for k in e:
            if isinstance(e[k], float):
                if abs(g[k] - e[k]) > 1e-6:
                    return False, f"seg[{i}].{k}: {g[k]} != {e[k]}"
            else:
                if g[k] != e[k]:
                    return False, f"seg[{i}].{k}: {g[k]!r} != {e[k]!r}"
    return True, "ok"


def test_pick_node():
    # spot-check the table
    assert pick_node("cooking", "living") == 4
    assert pick_node("dishwashing", "living") == 4
    assert pick_node("eating", "living") == 4
    assert pick_node("sleeping", "living") == 1
    assert pick_node("watching_tv", "living") == 1
    assert pick_node("other", "living") == 1
    assert pick_node("absence", "living") == 1
    assert pick_node("other", "hall") == 1
    assert pick_node("showering", "bathroom") == 13
    assert pick_node("toilet", "wcroom") == 11
    assert pick_node("sleeping", "bedroom") == 9
    print("test_pick_node: OK")


def test_example1_same_class_later_wins():
    """A: cooking:living [100,300]; B: dressing:bedroom [200,400]
    Both Class A — later (B) preempts earlier (A) at t=200.
    Expected: cooking [100,200], dressing [200,400]."""
    annots = [
        {"label": "cooking:living", "start": 0, "end": 300},
        {"label": "dressing:bedroom", "start": 200, "end": 400},
    ]
    annotations, node_trace = clean_annotations(annots)
    ok, msg = _approx(
        annotations,
        [
            {"start": 0.0, "end": 200.0, "event": "cooking"},
            {"start": 200.0, "end": 400.0, "event": "dressing"},
        ],
    )
    assert ok, f"annotations wrong: {msg}\nGot: {annotations}"
    ok, msg = _approx(
        node_trace,
        [
            {"start": 0.0, "end": 200.0, "node": "Node4"},  # cooking@living
            {"start": 200.0, "end": 400.0, "node": "Node9"},  # dressing@bedroom
        ],
    )
    assert ok, f"node_trace wrong: {msg}\nGot: {node_trace}"
    print("test_example1_same_class_later_wins: OK")


def test_example2_other_preempted_by_specific():
    """A: other:living [100,300]; B: dressing:bedroom [200,400]
    B is Class A, A is Class B — B preempts A at t=200.
    Expected: other:living [100,200], dressing [200,400]."""
    annots = [
        {"label": "other:living", "start": 0, "end": 300},
        {"label": "dressing:bedroom", "start": 200, "end": 400},
    ]
    annotations, node_trace = clean_annotations(annots)
    ok, msg = _approx(
        annotations,
        [
            {"start": 0.0, "end": 200.0, "event": "other"},
            {"start": 200.0, "end": 400.0, "event": "dressing"},
        ],
    )
    assert ok, f"annotations wrong: {msg}\nGot: {annotations}"
    ok, msg = _approx(
        node_trace,
        [
            {"start": 0.0, "end": 200.0, "node": "Node1"},
            {"start": 200.0, "end": 400.0, "node": "Node9"},
        ],
    )
    assert ok, f"node_trace wrong: {msg}\nGot: {node_trace}"
    print("test_example2_other_preempted_by_specific: OK")


def test_example3_specific_keeps_other_waits():
    """A: cooking:living [100,300]; B: other:living [200,400]
    A is Class A, B is Class B — A keeps going, B starts after A ends.
    Expected: cooking [100,300], other [300,400]."""
    annots = [
        {"label": "cooking:living", "start": 0, "end": 300},
        {"label": "other:living", "start": 200, "end": 400},
    ]
    annotations, node_trace = clean_annotations(annots)
    ok, msg = _approx(
        annotations,
        [
            {"start": 0.0, "end": 300.0, "event": "cooking"},
            {"start": 300.0, "end": 400.0, "event": "other"},
        ],
    )
    assert ok, f"annotations wrong: {msg}\nGot: {annotations}"
    ok, msg = _approx(
        node_trace,
        [
            {"start": 0.0, "end": 300.0, "node": "Node4"},  # cooking@living
            {"start": 300.0, "end": 400.0, "node": "Node1"},  # other@living
        ],
    )
    assert ok, f"node_trace wrong: {msg}\nGot: {node_trace}"
    print("test_example3_specific_keeps_other_waits: OK")


def test_example4_sandwich():
    """A: other:living [100,400] (Class B); B: cooking:bedroom [200,300] (Class A)
    B preempts A in the middle. Expected:
        other [100,200], cooking [200,300], other [300,400]"""
    annots = [
        {"label": "other:living", "start": 0, "end": 400},
        {"label": "cooking:bedroom", "start": 200, "end": 300},
    ]
    annotations, node_trace = clean_annotations(annots)
    ok, msg = _approx(
        annotations,
        [
            {"start": 0.0, "end": 200.0, "event": "other"},
            {"start": 200.0, "end": 300.0, "event": "cooking"},
            {"start": 300.0, "end": 400.0, "event": "other"},
        ],
    )
    assert ok, f"annotations wrong: {msg}\nGot: {annotations}"
    # cooking@bedroom — not in LIVING_KITCHEN exception, falls to bedroom rule → Node 9
    ok, msg = _approx(
        node_trace,
        [
            {"start": 0.0, "end": 200.0, "node": "Node1"},
            {"start": 200.0, "end": 300.0, "node": "Node9"},
            {"start": 300.0, "end": 400.0, "node": "Node1"},
        ],
    )
    assert ok, f"node_trace wrong: {msg}\nGot: {node_trace}"
    print("test_example4_sandwich: OK")


def test_all_absent_defaults_to_node1():
    """Multiple absence events covering the same range → output absence on Node1."""
    annots = [
        {"label": "absence:living", "start": 0, "end": 100},
        {"label": "absence:bedroom", "start": 0, "end": 100},
        {"label": "absence:bathroom", "start": 0, "end": 100},
    ]
    annotations, node_trace = clean_annotations(annots)
    assert annotations == [{"start": 0.0, "end": 100.0, "event": "absence"}], annotations
    assert node_trace == [{"start": 0.0, "end": 100.0, "node": "Node1"}], node_trace
    print("test_all_absent_defaults_to_node1: OK")


def test_merge_same_activity_diff_room():
    """cooking:living [100,200] then cooking:bedroom [200,300] → ONE annotation
    'cooking' [100,300] but TWO node_trace spans (Node4 then Node9)."""
    annots = [
        {"label": "cooking:living", "start": 0, "end": 200},
        {"label": "cooking:bedroom", "start": 200, "end": 300},
    ]
    annotations, node_trace = clean_annotations(annots)
    assert annotations == [{"start": 0.0, "end": 300.0, "event": "cooking"}], annotations
    assert node_trace == [
        {"start": 0.0, "end": 200.0, "node": "Node4"},
        {"start": 200.0, "end": 300.0, "node": "Node9"},
    ], node_trace
    print("test_merge_same_activity_diff_room: OK")


def test_drop_short_sliver_spans():
    """Sub-1s sliver between two real events gets absorbed into the
    previous span. Concretely: an absence event nested inside two
    overlapping bedroom-events creates a tiny gap when the earlier ends —
    the resulting <1s sliver should attach to the prior span.
    """
    annots = [
        {"label": "absence:living", "start": 0, "end": 100},
        {"label": "sleeping:bedroom", "start": 50, "end": 100.02},
        {"label": "dressing:bedroom", "start": 100.02, "end": 200},
    ]
    annotations, node_trace = clean_annotations(annots)
    # Sweep-line produces:
    #   absence [0,50]  (only absence covers)
    #   sleeping [50,100]  (sleeping=A wins over absence=C)
    #   absence [100,100.02]  (sliver, 0.02s)
    #   dressing [100.02,200]  (only dressing covers)
    # Drop-short collapses the 0.02s sliver into sleeping → 100.02.
    # Result: absence [0,50], sleeping [50,100.02], dressing [100.02,200]
    assert annotations == [
        {"start": 0.0, "end": 50.0, "event": "absence"},
        {"start": 50.0, "end": 100.02, "event": "sleeping"},
        {"start": 100.02, "end": 200.0, "event": "dressing"},
    ], annotations
    print("test_drop_short_sliver_spans: OK (sliver absorbed into prior)")


def test_leading_gap_prepended_as_other_on_node1():
    """If first event doesn't start at t=0, the leading gap should be
    prepended as 'other' @ Node 1 (not silence, not absence)."""
    annots = [
        {"label": "sleeping:bedroom", "start": 100, "end": 200},
    ]
    annotations, node_trace = clean_annotations(annots)
    assert annotations == [
        {"start": 0.0, "end": 100.0, "event": "other"},
        {"start": 100.0, "end": 200.0, "event": "sleeping"},
    ], annotations
    assert node_trace == [
        {"start": 0.0, "end": 100.0, "node": "Node1"},
        {"start": 100.0, "end": 200.0, "node": "Node9"},
    ], node_trace
    print("test_leading_gap_prepended_as_other_on_node1: OK")


def test_no_leading_gap_when_starts_at_zero():
    """If first event already starts at 0, don't prepend anything."""
    annots = [
        {"label": "sleeping:bedroom", "start": 0, "end": 200},
    ]
    annotations, node_trace = clean_annotations(annots)
    assert annotations == [
        {"start": 0.0, "end": 200.0, "event": "sleeping"},
    ], annotations
    print("test_no_leading_gap_when_starts_at_zero: OK")


def test_drop_short_explicit_sliver():
    """Construct an explicit sliver: tail of an event extends 0.5s past
    a higher-priority event's end, with nothing else to cover it —
    that tail should be absorbed into the preceding higher-priority span.
    """
    # A specific event covers [0, 100]. An "other" event covers [0, 100.5].
    # After A ends at 100, "other" should win [100, 100.5] alone (0.5s, < 1s).
    # That 0.5s "other" span should be dropped and the preceding "cooking"
    # span extended to 100.5.
    annots = [
        {"label": "cooking:living", "start": 0, "end": 100},
        {"label": "other:living", "start": 0, "end": 100.5},
    ]
    annotations, node_trace = clean_annotations(annots)
    # Expect: cooking [0, 100.5] (sliver absorbed)
    assert annotations == [
        {"start": 0.0, "end": 100.5, "event": "cooking"},
    ], annotations
    assert node_trace == [
        {"start": 0.0, "end": 100.5, "node": "Node4"},
    ], node_trace
    print("test_drop_short_explicit_sliver: OK")


def test_activity_label_normalized_to_underscore():
    """SINS_LABELS uses underscores ('watching_tv', 'getting_dry') — the
    mono builder must normalize incoming space-separated activity strings
    so the manifest matches the schema."""
    annots = [
        {"label": "watching tv:living", "start": 0.0, "end": 60.0},
        {"label": "getting dry:bathroom", "start": 60.0, "end": 120.0},
        {"label": "COOKING:living", "start": 120.0, "end": 180.0},  # case too
    ]
    annotations, _node_trace = clean_annotations(annots)
    activities = [a["event"] for a in annotations]
    assert activities == ["watching_tv", "getting_dry", "cooking"], activities
    # No spaces should ever appear in any emitted activity string.
    for a in annotations:
        assert " " not in a["event"], a
    print("test_activity_label_normalized_to_underscore: OK")


if __name__ == "__main__":
    test_pick_node()
    test_example1_same_class_later_wins()
    test_example2_other_preempted_by_specific()
    test_example3_specific_keeps_other_waits()
    test_example4_sandwich()
    test_all_absent_defaults_to_node1()
    test_merge_same_activity_diff_room()
    test_drop_short_sliver_spans()
    test_leading_gap_prepended_as_other_on_node1()
    test_no_leading_gap_when_starts_at_zero()
    test_drop_short_explicit_sliver()
    test_activity_label_normalized_to_underscore()
    print("\nALL TESTS PASSED")
