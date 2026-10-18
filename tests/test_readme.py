from pathlib import Path
import re


def test_readme_code_blocks_executable():
    readme_path = Path(__file__).parent.parent / "README.md"
    assert readme_path.exists(), "README.md must exist at repo root"

    content = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(r"```python\s+(.*?)```", re.DOTALL)
    code_blocks = pattern.findall(content)

    assert len(code_blocks) >= 6, f"Expected at least 6 runnable python code blocks, found {len(code_blocks)}"

    for i, code in enumerate(code_blocks):
        namespace = {}
        try:
            exec(code, namespace)
        except Exception as exc:
            raise AssertionError(f"Code block {i + 1} in README.md failed to execute:\n{code}\nError: {exc}") from exc
