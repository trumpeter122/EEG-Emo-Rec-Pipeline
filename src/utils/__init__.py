from rich.progress import BarColumn, Console, Progress, TextColumn, TimeElapsedColumn


def track(iterable, description: str, context: str):
    console = Console()
    console.print(
        f"[bold yellow]--{context}--[/bold yellow]\n"
        f"[bold blue]{description}[/bold blue]"
    )

    with Progress(
        TextColumn(""),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(description, total=len(iterable))
        for item in iterable:
            yield item
            progress.advance(task)
