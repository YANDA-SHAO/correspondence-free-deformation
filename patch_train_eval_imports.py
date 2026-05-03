from pathlib import Path

SRC = Path("src")


def remove_top_level_block(text: str, name: str, kind: str) -> str:
    """
    Remove a top-level function or class block.
    kind: "def" or "class"
    """
    lines = text.splitlines()
    start = None

    prefix = f"{kind} {name}"
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            start = i
            break

    if start is None:
        print(f"[skip] {kind} {name} not found")
        return text

    end = len(lines)
    for j in range(start + 1, len(lines)):
        line = lines[j]
        if (
            line.startswith("def ")
            or line.startswith("class ")
            or line.startswith("if __name__")
        ):
            end = j
            break

    print(f"[remove] {kind} {name}: lines {start+1}-{end}")
    new_lines = lines[:start] + lines[end:]
    return "\n".join(new_lines) + "\n"


def add_imports(text: str, import_block: str) -> str:
    if import_block.strip() in text:
        print("[skip] imports already exist")
        return text

    # Put imports after module docstring if possible
    lines = text.splitlines()
    if len(lines) > 1 and lines[1].startswith('"""'):
        end_doc = None
        for i in range(2, len(lines)):
            if lines[i].startswith('"""'):
                end_doc = i
                break
        if end_doc is not None:
            new_lines = (
                lines[: end_doc + 1]
                + [""]
                + import_block.strip().splitlines()
                + [""]
                + lines[end_doc + 1 :]
            )
            return "\n".join(new_lines) + "\n"

    return import_block.strip() + "\n\n" + text


def patch_file(path: Path, import_block: str, removals):
    text = path.read_text(encoding="utf-8")

    backup = path.with_suffix(path.suffix + ".before_modular_imports.bak")
    backup.write_text(text, encoding="utf-8")
    print(f"[backup] {backup}")

    text = add_imports(text, import_block)

    for kind, name in removals:
        text = remove_top_level_block(text, name, kind)

    path.write_text(text, encoding="utf-8")
    print(f"[write] {path}")


train_imports = """
from models.joint_uw_model import JointUWProbabilisticMLP
from geometry.observation import observation_operator_softW, observation_operator_gtW
from losses.loss_core import kl_loss, gaussian_nll_loss, mse_loss, soft_ce_loss_from_sparse
"""

eval_imports = """
from models.joint_uw_model import JointUWProbabilisticMLP
from geometry.observation import observation_operator_softW
"""

patch_file(
    SRC / "train.py",
    train_imports,
    removals=[
        ("def", "kl_loss"),
        ("def", "gaussian_nll_loss"),
        ("def", "mse_loss"),
        ("def", "soft_ce_loss_from_sparse"),
        ("def", "torch_perspective_project"),
        ("def", "observation_operator_softW"),
        ("def", "observation_operator_gtW"),
        ("class", "JointUWProbabilisticMLP"),
    ],
)

patch_file(
    SRC / "eval.py",
    eval_imports,
    removals=[
        ("def", "torch_perspective_project"),
        ("def", "observation_operator_softW"),
        ("class", "JointUWProbabilisticMLP"),
    ],
)

print("Done.")