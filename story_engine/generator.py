"""
StoryAI V8 Generator

Generates story chapters using the self-trained V8 GPT model.
Compatible with StoryManager and StoryAI V10 NLU.
"""

import hashlib
from pathlib import Path
from typing import Any

import torch

from model import GPT, GPTConfig
from tokenizer import BPETokenizer
from generate_novel import sample


class StoryGenerator:
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/storyai_v8_best.pt",
        tokenizer_path: str = "dataset/processed_structured/tokenizer.json",
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        checkpoint_path = Path(checkpoint_path)
        tokenizer_path = Path(tokenizer_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"V8 checkpoint not found: {checkpoint_path}"
            )

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"V8 tokenizer not found: {tokenizer_path}"
            )

        print(f"Loading StoryAI V8 on {self.device}...")

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        # Verify tokenizer compatibility
        expected_hash = checkpoint.get("tokenizer_sha256")

        if expected_hash:
            actual_hash = hashlib.sha256(
                tokenizer_path.read_bytes()
            ).hexdigest()

            if actual_hash != expected_hash:
                raise ValueError(
                    "Tokenizer mismatch: the tokenizer does not "
                    "match the V8 training checkpoint."
                )

        # Load tokenizer
        self.tokenizer = BPETokenizer.load(tokenizer_path)

        # Initialize the exact V8 model architecture
        config = GPTConfig(**checkpoint["config"])

        self.model = GPT(config).to(self.device)

        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        del checkpoint

        print("StoryAI V8 generator loaded successfully.")

    def _format_characters(
        self,
        story: dict[str, Any],
    ) -> str:
        """Convert story characters into generation context."""

        formatted = []

        for character in story.get("characters", []):
            details = [
                f"Name: {character.get('name', '')}",
                f"Role: {character.get('role', 'character')}",
            ]

            if character.get("story_position"):
                details.append(
                    f"Story position: {character['story_position']}"
                )

            if character.get("description"):
                details.append(
                    f"Description: {character['description']}"
                )

            if character.get("personality"):
                personality = character["personality"]

                if isinstance(personality, list):
                    personality = ", ".join(personality)

                details.append(f"Personality: {personality}")

            if character.get("abilities"):
                abilities = character["abilities"]

                if isinstance(abilities, list):
                    abilities = ", ".join(abilities)

                details.append(f"Abilities: {abilities}")

            relationships = character.get("relationships", [])

            for relationship in relationships:
                details.append(
                    "Relationship: "
                    f"{relationship.get('character', '')} "
                    f"({relationship.get('type', '')})"
                )

            formatted.append("; ".join(details))

        return "\n".join(formatted)

    def _get_previous_context(
        self,
        story: dict[str, Any],
        max_chars: int = 400,
    ) -> str:
        """Get recent chapter context for continuity."""

        chapters = story.get("chapters", [])

        if not chapters:
            return ""

        previous_content = []

        for chapter in chapters[-2:]:
            content = chapter.get("content", "")

            if content:
                previous_content.append(content)

        combined = "\n".join(previous_content)

        return combined[-max_chars:]

    def _build_prompt(
        self,
        story: dict[str, Any],
        chapter_number: int,
        instruction: str,
    ) -> str:
        """Build a structured prompt for the V8 model."""

        title = story.get("title", "Untitled")

        premise = (
            story.get("premise")
            or story.get("plot")
            or f"A story titled {title}."
        )

        characters = self._format_characters(story)

        previous = self._get_previous_context(story)

        sections = [
            "<INSTRUCTION>",
            "Write a story chapter as narrative prose.",
            f"Title: {title}",
            f"Premise: {premise}",
        ]

        if story.get("plot"):
            sections.append(f"Current plot: {story['plot']}")

        if characters:
            sections.append(f"Characters:\n{characters}")

        sections.append(
            f"Chapter {chapter_number}: {instruction}"
        )

        if previous:
            sections.append(f"Previously:\n{previous}")

        sections.extend([
            "Write only the scene, not character profiles or metadata.",
            "<STORY>",
            "",
        ])

        return "\n".join(sections)

    def generate_chapter(
        self,
        story: dict[str, Any],
        chapter_number: int,
        instruction: str,
        tokens: int = 300,
        temperature: float = 0.7,
        top_k: int = 20,
        repetition_penalty: float = 1.12,
    ) -> str:
        """
        Generate chapter content using StoryAI V8.

        Returns generated text without modifying the story.
        """

        if chapter_number < 1:
            raise ValueError(
                "Chapter number must be greater than zero."
            )

        if not instruction.strip():
            raise ValueError(
                "Chapter instruction cannot be empty."
            )

        if tokens < 1:
            raise ValueError(
                "Token count must be greater than zero."
            )

        if temperature <= 0:
            raise ValueError(
                "Temperature must be greater than zero."
            )

        if repetition_penalty < 1:
            raise ValueError(
                "Repetition penalty must be at least 1."
            )

        prompt = self._build_prompt(
            story=story,
            chapter_number=chapter_number,
            instruction=instruction,
        )

        print(
            f"Generating chapter {chapter_number} "
            f"({tokens} tokens) on {self.device}..."
        )

        generated_text = sample(
            model=self.model,
            tok=self.tokenizer,
            prompt=prompt,
            n_tokens=tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        # Remove any special instruction markers from output
        for marker in (
            "<INSTRUCTION>",
            "<STORY>",
            "<END_STORY>",
        ):
            generated_text = generated_text.replace(marker, "")

        return generated_text.strip()