import json
import typer
from rich import print as rprint
from rich.table import Table

from ..config import settings

app = typer.Typer(name="model", help="Manage model profiles for Pydantic AI. (list/add/show/remove)")


COMMON_KEYS = {
    "max_tokens",
    "temperature",
    "top_p",
    "timeout",
    "parallel_tool_calls",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "stop_sequences",
    "extra_headers",
    "extra_body",
}

PROVIDER_PREFIXES = {"openai", "anthropic", "bedrock", "cohere", "google", "grok", "huggingface"}


def _is_valid_setting_key(key: str) -> bool:
    """Heuristic validation for model settings.

    - True for common cross-provider keys supported by Pydantic AI.
    - True if key starts with a known provider prefix (case-insensitive), e.g. "openai_".
    - False otherwise. False does not block; callers should warn but still accept.
    """
    if key in COMMON_KEYS:
        return True
    kl = key.lower()
    return any(kl.startswith(p + "_") or kl.startswith(p + ":") or kl.startswith(p)
               for p in PROVIDER_PREFIXES)


def _parse_value(raw: str):
    """Parse a JSON-ish value from user input.

    Accepts JSON, otherwise falls back to str. Convenient bool/number parsing included.
    """
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        # Fallback for simple bools without JSON casing
        if raw.lower() in {"true", "false"}:
            return raw.lower() == "true"
        # Try number
        try:
            if "." in raw:
                return float(raw)
            return int(raw)
        except Exception:
            return raw


@app.command("list")
def list_models():
    """List all model profiles"""
    table = Table()
    table.add_column("Name", justify="left")
    table.add_column("Model", justify="left", overflow="fold")
    table.add_column("Settings (keys)", justify="left", overflow="fold")

    for name, prof in settings.models.items():
        model = prof.get("model", "-")
        keys = ", ".join(sorted((prof.get("settings") or {}).keys()))
        table.add_row(name, model, keys)

    rprint(table)


@app.command("add")
def add_model(
    name: str = typer.Argument(None, help="Name of the model profile"),
    model: str = typer.Option(None, "--model", "-m", help="Pydantic AI model, e.g. openai:gpt-4o-mini"),
    set_item: list[str] = typer.Option(
        None,
        "--set",
        help="Model setting as key=JSON_value (repeatable)",
    ),
):
    """Add a new model profile. Interactive by default, flags supported."""
    if not name:
        name = typer.prompt("Enter name")

    confirm = True
    if name in settings.models:
        confirm = typer.confirm(f"Model profile {name} already exists. Overwrite it?")

    if not model:
        model = typer.prompt("Enter model (e.g., openai:gpt-4o-mini)")
    if ":" not in model:
        rprint("[red]Invalid model. Use provider:model syntax (e.g., openai:gpt-5-mini)[/red]")
        return

    # Collect settings either from flags or interactively
    settings_dict: dict = {}
    # From flags
    if set_item:
        for item in set_item:
            if "=" not in item:
                rprint("[red]Invalid --set format. Use key=JSON_value[/red]")
                return
            key, raw_val = item.split("=", 1)
            key = key.strip()
            if not _is_valid_setting_key(key):
                rprint(f"[yellow]Warning:[/] setting key may not be recognized: {key}")
            settings_dict[key] = _parse_value(raw_val)

    # Interactive loop
    while True:
        entry = typer.prompt(
            "Enter setting (key=JSON_value, leave empty to finish)", default=""
        )
        if not entry:
            break
        if "=" not in entry:
            rprint("[red]Invalid format. Use key=JSON_value[/red]")
            continue
        key, raw_val = entry.split("=", 1)
        key = key.strip()
        if not _is_valid_setting_key(key):
            rprint(
                "[yellow]Warning:[/] setting key may not be recognized: "
                + key
            )
        settings_dict[key] = _parse_value(raw_val)

    if confirm:
        settings.models[name] = {
            "model": model,
            "settings": settings_dict,
        }
        settings.save()
        rprint(f"Added model profile {name}")


@app.command("remove")
def remove_model(name: str = typer.Argument(None, help="Name of the model profile")):
    """Remove an existing model profile"""
    if not name:
        name = typer.prompt("Enter name")
    if name in settings.models:
        del settings.models[name]
        settings.save()
        rprint(f"Deleted model profile {name}")
    else:
        rprint(f"No model profile found with name {name}")
