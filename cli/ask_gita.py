"""
CLI for the Gita SLM.

Usage:
    # Interactive mode (prompt loop):
    python cli/ask_gita.py

    # Single verse from command line:
    python cli/ask_gita.py "karmanye vadhikaraste ma phaleshu kadachana"

    # With options:
    python cli/ask_gita.py --verse "yada yada hi dharmasya" --lang en --temp 0.7
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.prompt import Prompt
from rich.rule import Rule

from config import TEMPERATURE, TOP_K, MAX_NEW_TOKENS
from src.inference.generate import get_engine

console = Console()

BANNER = """
[bold saffron]  ॐ  Gita SLM — Bhagavad Gita Verse Interpreter  ॐ[/bold saffron]
[dim]  A Small Language Model trained on the Bhagavad Gita[/dim]
[dim]  Built from scratch: transformer architecture, BPE tokenizer, AdamW training[/dim]
"""

SAMPLE_VERSES = [
    "karmanye vadhikaraste ma phaleshu kadachana",
    "yada yada hi dharmasya glanir bhavati bharata",
    "na jayate mriyate va kadachin",
    "uddhared atmanatmanam natmanam avasadayet",
    "sarva-dharman parityajya mam ekam sharanam vraja",
]


def display_result(result: dict, language: str):
    """Pretty-prints the generated meaning."""

    # Verse panel
    console.print(
        Panel(
            f"[bold cyan]{result['verse']}[/bold cyan]",
            title="[bold]📿 Shloka[/bold]",
            border_style="cyan",
        )
    )

    # English meaning
    if language in ("en", "both") and result["meaning_en"]:
        console.print(
            Panel(
                f"[white]{result['meaning_en']}[/white]",
                title="[bold green]🇬🇧 English Meaning[/bold green]",
                border_style="green",
            )
        )

    # Hindi meaning
    if language in ("hi", "both") and result["meaning_hi"]:
        console.print(
            Panel(
                f"[yellow]{result['meaning_hi']}[/yellow]",
                title="[bold yellow]🇮🇳 Hindi Meaning (हिंदी अर्थ)[/bold yellow]",
                border_style="yellow",
            )
        )

    if not result["meaning_en"] and not result["meaning_hi"]:
        console.print(
            Panel(
                f"[dim]{result['raw']}[/dim]",
                title="[bold red]Raw Output[/bold red]",
                border_style="red",
            )
        )


def show_examples():
    """Shows sample verses the user can try."""
    table = Table(title="Sample Shlokas to Try", border_style="cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Verse (transliterated)", style="cyan")

    for i, v in enumerate(SAMPLE_VERSES, 1):
        table.add_row(str(i), v)

    console.print(table)


def run_interactive(engine, language: str, temperature: float, top_k: int):
    """Interactive REPL loop."""
    console.print(BANNER)
    show_examples()
    console.print(
        "\n[dim]Type a verse shloka, 'examples' to see samples, or 'quit' to exit.[/dim]\n"
    )

    while True:
        try:
            verse = Prompt.ask("[bold saffron]Enter shloka[/bold saffron]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Namaste 🙏[/dim]")
            break

        if not verse:
            continue

        if verse.lower() in ("quit", "exit", "q"):
            console.print("[dim]Namaste 🙏[/dim]")
            break

        if verse.lower() == "examples":
            show_examples()
            continue

        with console.status("[bold cyan]Generating meaning...[/bold cyan]"):
            result = engine.generate_meaning(
                verse=verse,
                temperature=temperature,
                top_k=top_k,
                language=language,
            )

        display_result(result, language)
        console.print()


def run_single(engine, verse: str, language: str, temperature: float, top_k: int):
    """Generate meaning for a single verse and exit."""
    with console.status("[bold cyan]Generating meaning...[/bold cyan]"):
        result = engine.generate_meaning(
            verse=verse,
            temperature=temperature,
            top_k=top_k,
            language=language,
        )

    display_result(result, language)


def main():
    parser = argparse.ArgumentParser(
        description="Gita SLM — Ask the Bhagavad Gita SLM for verse meanings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "verse",
        nargs="?",
        default=None,
        help="Verse text (if omitted, enters interactive mode)",
    )
    parser.add_argument(
        "--lang", "-l",
        choices=["en", "hi", "both"],
        default="both",
        help="Output language (default: both)",
    )
    parser.add_argument(
        "--temp", "-t",
        type=float,
        default=TEMPERATURE,
        help=f"Sampling temperature (default: {TEMPERATURE})",
    )
    parser.add_argument(
        "--topk", "-k",
        type=int,
        default=TOP_K,
        help=f"Top-k sampling (default: {TOP_K})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help=f"Max tokens to generate (default: {MAX_NEW_TOKENS})",
    )

    args = parser.parse_args()

    # Load model
    console.print("[dim]Loading Gita SLM...[/dim]")
    try:
        engine = get_engine()
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    console.print(Rule(style="saffron"))

    if args.verse:
        run_single(engine, args.verse, args.lang, args.temp, args.topk)
    else:
        run_interactive(engine, args.lang, args.temp, args.topk)


if __name__ == "__main__":
    main()
