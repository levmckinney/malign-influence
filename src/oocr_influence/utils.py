import subprocess
import os
from pathlib import Path


def get_root_of_git_repo(path: Path | str = ".") -> str:
    """
    Get the root directory of the git repository at the given path.

    Args:
        path: A path within a git repository

    Returns:
        The absolute path to the root of the git repository

    Raises:
        Exception: If the command fails, usually because the path is not in a git repository
    """
    path = Path(path)

    abs_path = path.absolute()
    current_dir = abs_path.parent if abs_path.is_file() else abs_path
    command = ["git", "-C", current_dir, "rev-parse", "--show-toplevel"]

    result = subprocess.run(command, capture_output=True, text=True, check=True)

    if result.returncode != 0:
        raise Exception(
            f"Failed to get git root for path: {path}, command: {' '.join(command)}, stdout: {result.stdout}, stderr: {result.stderr}"
        )

    return result.stdout.strip()
