"""Shared, deliberately explicit planning-task format for Story AI v16."""
import json

INSTRUCTION = (
    "Read the fiction context. Return a JSON story plan grounded in the context. "
    "Preserve characters and established facts; do not invent prior events."
)

def prompt_for(context):
    return f"{INSTRUCTION}\n\nStory so far:\n{context.strip()}\n\nStory plan JSON:\n"

def target_for(row):
    return json.dumps({
        "characters": row["characters"],
        "facts": [f["fact"] for f in row["facts"]],
        "next_event": row["plan"]["next_event"],
        "beats": row["plan"]["beats"],
        "constraints": row["plan"]["constraints"],
    }, ensure_ascii=False, separators=(",", ":"))

def validate_row(row):
    required = {"story_id", "context", "characters", "facts", "plan"}
    if not isinstance(row, dict) or not required.issubset(row):
        raise ValueError("Missing required row fields: " + ", ".join(sorted(required - set(row) if isinstance(row,dict) else required)))
    if not isinstance(row["story_id"], str) or not row["story_id"].strip():
        raise ValueError("story_id must be nonempty")
    context = row["context"]
    if not isinstance(context, str) or len(context.strip()) < 80:
        raise ValueError("context must have at least 80 characters")
    chars = row["characters"]
    if not isinstance(chars, list) or not chars or not all(isinstance(c,str) and c.strip() and c in context for c in chars):
        raise ValueError("characters must be nonempty and literally present in context")
    facts = row["facts"]
    if not isinstance(facts, list) or not facts:
        raise ValueError("at least one fact with an evidence quote is required")
    for f in facts:
        if not isinstance(f, dict) or not isinstance(f.get("fact"),str) or not f["fact"].strip():
            raise ValueError("each fact requires nonempty fact")
        if not isinstance(f.get("evidence"),str) or not f["evidence"].strip() or f["evidence"] not in context:
            raise ValueError("each fact requires verbatim evidence substring from context")
    plan = row["plan"]
    if not isinstance(plan,dict) or not isinstance(plan.get("next_event"),str) or not plan["next_event"].strip():
        raise ValueError("plan.next_event required")
    for key in ("beats", "constraints"):
        v = plan.get(key)
        if not isinstance(v,list) or not v or not all(isinstance(x,str) and x.strip() for x in v):
            raise ValueError(f"plan.{key} requires nonempty string list")
    if row.get("reviewed") is not True:
        raise ValueError("row must have reviewed=true after human review; evidence matching is not semantic verification")
    return True
