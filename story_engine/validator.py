from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    valid: bool
    score: float
    issues: list[str] = field(default_factory=list)


class ChapterValidator:

    def validate(
        self,
        content: str,
        required_characters: list[str] | None = None,
        required_keywords: list[str] | None = None,
        min_words: int = 50,
    ) -> ValidationResult:

        issues = []

        text = content.strip()

        if not text:
            return ValidationResult(
                valid=False,
                score=0.0,
                issues=["Generated chapter is empty."],
            )

        words = text.split()

        if len(words) < min_words:
            issues.append(
                f"Chapter is too short: {len(words)} words."
            )

        text_lower = text.casefold()

        required_characters = required_characters or []

        for name in required_characters:
            if name.casefold() not in text_lower:
                issues.append(
                    f"Missing required character: {name}"
                )

        required_keywords = required_keywords or []

        for keyword in required_keywords:
            if keyword.casefold() not in text_lower:
                issues.append(
                    f"Missing requested element: {keyword}"
                )

        # Detect excessive repetition.
        if len(words) >= 20:
            normalized = [word.casefold() for word in words]

            unique_ratio = (
                len(set(normalized)) / len(normalized)
            )

            if unique_ratio < 0.25:
                issues.append(
                    "Excessive word repetition detected."
                )

        # Detect obviously incomplete endings.
        if text[-1] not in ".!?\"'”’":
            issues.append(
                "Chapter appears to end mid-sentence."
            )

        score = max(
            0.0,
            1.0 - len(issues) * 0.2,
        )

        return ValidationResult(
            valid=len(issues) == 0,
            score=score,
            issues=issues,
        )