import os
import subprocess
from typing import Optional, Dict, Any


def clone_repo(repo_url: str, dest_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Clone a git repository using the system `git` command.

    Args:
        repo_url: Repository URL to clone.
        dest_path: Destination path. If None, derives directory name from URL.

    Returns:
        Dict with keys: `success`(bool), `path`(str or None), `output`(str).
    """
    try:
        if dest_path is None:
            repo_name = os.path.splitext(os.path.basename(repo_url.rstrip('/')))[0]
            dest_path = repo_name or "repo_clone"

        # Ensure parent directory exists
        parent = os.path.dirname(dest_path) or "."
        os.makedirs(parent, exist_ok=True)

        # Run git clone
        proc = subprocess.run(["git", "clone", repo_url, dest_path], capture_output=True, text=True)
        output = proc.stdout + proc.stderr
        success = proc.returncode == 0

        return {"success": success, "path": dest_path if success else None, "output": output}
    except Exception as e:
        return {"success": False, "path": None, "output": str(e)}


if __name__ == '__main__':
    # Simple local test
    import sys
    if len(sys.argv) < 2:
        print("Usage: python git_cloner.py <repo_url> [dest_path]")
    else:
        url = sys.argv[1]
        dest = sys.argv[2] if len(sys.argv) > 2 else None
        print(clone_repo(url, dest))
