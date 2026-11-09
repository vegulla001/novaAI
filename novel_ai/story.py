from .llm import GenerationError
from .continuity import audit
from .prompts import plan_prompt, chapter_prompt, memory_prompt, critique_prompt

EMPTY_MEMORY = {"characters": {}, "locations": {}, "chronology": [],
                "unresolved_threads": [], "items": {}, "facts": []}

def validate_plan(plan: dict, chapters: int) -> dict:
    if not isinstance(plan.get("title"), str) or not plan["title"].strip():
        raise GenerationError("Plan requires title")
    if not isinstance(plan.get("chapters"), list) or len(plan["chapters"]) != chapters:
        raise GenerationError("Incorrect chapter plan count")
    if not isinstance(plan.get("characters"), list):
        raise GenerationError("Plan requires characters")
    numbers = [c.get("number") for c in plan["chapters"] if isinstance(c, dict)]
    if numbers != list(range(1, chapters + 1)):
        raise GenerationError("Invalid chapter numbers")
    for chapter in plan["chapters"]:
        for key in ["title", "objective", "conflict", "turning_point", "ending_hook"]:
            if not isinstance(chapter.get(key), str):
                raise GenerationError(f"Chapter plan missing {key}")
    return plan

def make_plan(llm, novel):
    return validate_plan(llm.structured(plan_prompt(novel), max_tokens=5000), novel.chapters_target)

def write_chapter(llm, novel, state, number, summaries, previous_tail):
    chapter = llm.generate(chapter_prompt(novel, state, number, summaries, previous_tail),
                           max_tokens=min(7500, max(750, int(novel.words_target * 2.1))))
    if len(chapter.split()) < max(90, int(novel.words_target * .25)):
        raise GenerationError("Chapter significantly shorter than requested")
    critique = llm.structured(critique_prompt(state, chapter, number), max_tokens=6000)
    score = critique.get("score", 0)
    if not isinstance(score, int) or not 0 <= score <= 10:
        raise GenerationError("Invalid critique score")
    replacement = critique.get("revised_text", "")
    if isinstance(replacement, str) and len(replacement.split()) >= len(chapter.split()) * .65:
        chapter = replacement
    violations = audit(chapter, state)
    if violations:
        raise GenerationError("Identity audit failed: " + "; ".join(violations[:5]))
    updated = llm.structured(memory_prompt(state, chapter, number), max_tokens=2500)
    summary, memory = updated.get("summary"), updated.get("memory")
    if not isinstance(summary, str) or not summary.strip() or not isinstance(memory, dict):
        raise GenerationError("Invalid story memory update")
    memory_violations = audit(summary + " " + __import__("json").dumps(memory, ensure_ascii=False), state)
    if memory_violations:
        raise GenerationError("Identity audit failed in updated memory: " + "; ".join(memory_violations[:5]))
    # Protect downstream prompt code from missing JSON fields.
    normalized = {**EMPTY_MEMORY, **memory}
    return chapter, summary.strip(), normalized, str(critique.get("issues", []))[:2000]
