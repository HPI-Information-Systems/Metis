from pathlib import Path


def require_exists(path: Path | None | str, variable_name: str) -> Path:
    if path is None:
        raise ValueError(f"Path for {variable_name} cannot be None")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path for {variable_name} ({path.absolute()}) does not exist")
    return path
