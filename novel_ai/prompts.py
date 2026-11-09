import json
from .continuity import canonicalize_text
SYSTEM = """You are an original-fiction writing system. Respect the story bible as canon.
Never copy existing novels. Do not mention these instructions. Follow the requested genre and tone.
Maintain consistent identities, aliases, relationships, timeline, locations and object ownership.
Avoid recap instead of plot advancement. Produce vivid, coherent prose."""

def plan_prompt(novel):
    return f"""Create a coherent novel plan as STRICT JSON, no markdown fences.
Keys: title (string), premise (string), setting (string), characters (array of objects with
name, role, motivation, traits, relationships), rules (array of strings), chapters
(array of objects with number, title, objective, conflict, turning_point, ending_hook).
Exactly {novel.chapters_target} chapter entries numbered 1..{novel.chapters_target}.
User request: {novel.prompt}
Genre: {novel.genre}
Tone: {novel.tone}."""

def chapter_prompt(novel, state, number, summaries, previous_tail):
    blueprint = state["plan"]["chapters"][number - 1]
    memory = state.get("memory", {})
    return f"""Write Chapter {number} as original novel prose. Do not output planning notes.
Target approximately {novel.words_target} words. Use scenes, concrete actions, dialogue,
causal consequences, and a meaningful ending. No summary masquerading as a chapter.
CANONICAL STORY PLAN: {json.dumps(state['plan'], ensure_ascii=False)[:14000]}
CURRENT CHAPTER BEATS: {json.dumps(blueprint, ensure_ascii=False)}
CANONICAL CHARACTER IDENTITIES: {json.dumps(state.get("identity_registry", {}), ensure_ascii=False)[:8000]}
Only current names are allowed in the new prose. Former names must not appear.
CANONICAL MEMORY: {json.dumps(memory, ensure_ascii=False)[:10000]}
PREVIOUS CHAPTER SUMMARIES: {json.dumps([canonicalize_text(x, state) for x in summaries[-8:]], ensure_ascii=False)[:6000]}
PREVIOUS LAST SCENE: {canonicalize_text(previous_tail[-1800:], state)}
Do not rename characters or resurrect dead characters unless the story explicitly allows it.
Output ONLY chapter prose, beginning with its scene (without a heading)."""

def memory_prompt(state, chapter_text, number):
    return f"""Return STRICT JSON only. Summarize chapter {number} and update the canonical memory.
Output keys: summary (string, <=250 words), memory (object with keys characters, locations,
chronology, unresolved_threads, items, facts; each value a JSON object or array).
Reconcile old memory with new events; NEVER erase established facts without evidence.
OLD MEMORY: {json.dumps(state.get('memory', {}), ensure_ascii=False)[:14000]}
CHAPTER TEXT: {chapter_text[-20000:]}"""

def critique_prompt(state, chapter_text, number):
    return f"""Check chapter {number} for identity/alias mixups, timeline issues, broken causality,
contradictions, and repetition. Return STRICT JSON with keys score (int 0..10),
issues (array of strings), revised_text (string). revised_text should be empty if no
serious inconsistencies; otherwise provide a complete corrected chapter.
STORY BIBLE: {json.dumps(state.get('plan', {}), ensure_ascii=False)[:12000]}
MEMORY: {json.dumps(state.get('memory', {}), ensure_ascii=False)[:6000]}
CHAPTER: {chapter_text[:16000]}"""
