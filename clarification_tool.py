#!/usr/bin/env python3
"""
Clarification Tool - Ask multiple questions at once, collect answers as JSON.
"""

import re
import sys
import json
from rich.console import Console
from rich.text import Text
from pydantic import BaseModel, Field
from langchain_core.tools import tool, Tool

console = Console()

# ── color palette (matches momobot's existing scheme) ─────────────────
C_ACCENT  = "#C8603A"   # coral — matches "You:" label
C_DIM     = "dim"
C_QUES    = "bold white"
C_COUNTER = "#AFA9EC"   # soft purple for the n/N counter
C_OK      = "#5DCAA5"   # teal for confirmations
C_RULE    = "#3a3a3a"   # subtle divider color


def _rule(width: int = 54) -> str:
    return "─" * width


def ask_clarifying_questions(questions: list[str]) -> dict:
    """
    Prompts the user with each question and returns a structured dict of answers.
    - Keys are question slugs
    - Empty answers stored as None (skipped)
    - Vague answers trigger one follow-up
    """

    def make_slug(text: str) -> str:
        words = re.sub(r'[^\w\s]', '', text.lower()).split()
        return '_'.join(words[:5])

    def is_vague(answer: str) -> bool:
        vague = {
            "idk", "not sure", "maybe", "dunno", "i don't know",
            "unsure", "no idea", "whatever", "anything", "don't care", "idc"
        }
        return answer.lower() in vague

    total = len(questions)

    # ── header ────────────────────────────────────────────────────────
    console.print()
    console.print(f"[{C_OK}]┌─ clarification needed {'─' * 29}┐[/{C_OK}]")
    console.print(f"[{C_OK}]│[/{C_OK}]  [{C_DIM}]{total} question{'s' if total != 1 else ''}  ·  press enter to skip[/{C_DIM}]")
    console.print(f"[{C_OK}]└{'─' * 52}┘[/{C_OK}]")
    console.print()

    responses = {}

    for i, question in enumerate(questions, 1):
        slug = make_slug(question)
        original_slug = slug
        counter = 2
        while slug in responses:
            slug = f"{original_slug}_{counter}"
            counter += 1

        # ── divider + question ────────────────────────────────────────
        console.print(f"  [{C_DIM}]{_rule()}[/{C_DIM}]")
        console.print(
            f"  [{C_COUNTER}]{i} / {total}[/{C_COUNTER}]  [{C_QUES}]{question}[/{C_QUES}]"
        )
        console.print(f"  [{C_DIM}]{_rule()}[/{C_DIM}]")

        raw = console.input(f"  [{C_ACCENT}]❯[/{C_ACCENT}] ").strip()

        if not raw:
            responses[slug] = None
            console.print(f"  [{C_DIM}]↳ skipped[/{C_DIM}]")
            console.print()
            continue

        if is_vague(raw):
            console.print(
                f"  [{C_DIM}]↳ that's a bit vague — elaborate? "
                f"(enter to keep '[italic]{raw}[/italic]')[/{C_DIM}]"
            )
            follow_up = console.input(f"  [{C_ACCENT}]❯[/{C_ACCENT}] ").strip()
            raw = follow_up if follow_up else raw

        responses[slug] = raw
        console.print(f"  [{C_OK}]✓[/{C_OK}] [{C_DIM}]saved[/{C_DIM}]")
        console.print()

    # ── summary footer ────────────────────────────────────────────────
    answered = sum(1 for v in responses.values() if v is not None)
    skipped  = len(responses) - answered

    console.print(f"  [{C_DIM}]{_rule()}[/{C_DIM}]")

    summary = Text("  ")
    summary.append("✓ ", style=C_OK)
    summary.append(f"{answered} answered", style="bold")
    if skipped:
        summary.append("  ·  ", style=C_DIM)
        summary.append(f"{skipped} skipped", style=C_DIM)
        skipped_keys = [k for k, v in responses.items() if v is None]
        summary.append(f"  ({', '.join(skipped_keys)})", style=C_DIM)
    console.print(summary)

    console.print(f"  [{C_DIM}]{_rule()}[/{C_DIM}]")
    console.print()

    return responses



class ClarifyingQuestionsInput(BaseModel):
    questions: list[str] = Field(
        description=(
            "A list of specific questions to ask the user. "
            "Each question should be a complete sentence. "
            "Example: ['What programming language do you prefer?', 'Should the output be a file or printed to console?']"
        )
    )
    
ask_clarifying_questions_tool = Tool(
    name="ask_clarifying_questions",
    func=lambda questions: ask_clarifying_questions(questions),
    description=(
        "Use this tool when you are uncertain about the user's preferences or requirements "
        "and need clarification before proceeding. Pass a list of clear, specific questions. "
        "The tool will prompt the user for each answer and return their responses as a dict "
        "like {'q1': 'answer1', 'q2': 'answer2'}. "
        "Use this BEFORE starting a task if key details are unclear."
    ),
    args_schema=ClarifyingQuestionsInput,
)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    if sys.argv[1] == "--interactive":
        print("Enter questions (one per line, empty line to finish):")
        questions = []
        while True:
            line = input().strip()
            if not line:
                break
            questions.append(line)
    else:
        with open(sys.argv[1], 'r') as f:
            questions = json.load(f)
    
    ask_clarifying_questions(questions)


if __name__ == "__main__":
    main()
