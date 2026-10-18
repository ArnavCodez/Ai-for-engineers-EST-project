"""Run one of our notebooks on Kaggle and copy what it produced back into this folder.

    python kaggle/run_on_kaggle.py 01_data_check
    python kaggle/run_on_kaggle.py 01_data_check --no-push   (wait for the run already on Kaggle and collect it)

Kaggle runs a short script that executes the notebook with nbconvert, so the copy that
comes back (notebooks/<name>.ipynb) has every output saved inside it.
Kernel settings (Kaggle slug, GPU, attached data) are in kaggle/kernels.json.
Needs the Kaggle CLI, signed in once with `kaggle auth login`.
"""
import base64
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RUNNER = '''import base64, pathlib, shutil, subprocess, sys
work = pathlib.Path("/kaggle/working/notebooks")
work.mkdir(parents=True, exist_ok=True)
notebook = work / "{name}.ipynb"
notebook.write_bytes(base64.b64decode("{source}"))
if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi"])
else:
    print("no NVIDIA GPU on this machine")
done = subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                       "--ExecutePreprocessor.timeout=-1", str(notebook)], cwd=work)
sys.exit(done.returncode)
'''


def kaggle(*args):
    result = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"kaggle {' '.join(args)} failed:\n{result.stderr or result.stdout}")
    return result.stdout


def signed_in_user():
    # Only the username line is read; the rest of `config view` is never printed.
    for line in kaggle("config", "view").splitlines():
        if "username" in line:
            return line.split(":", 1)[1].strip()
    sys.exit("Not signed in to Kaggle. Run `kaggle auth login` first.")


USER = signed_in_user()


def push(name, spec, build):
    shutil.rmtree(build, ignore_errors=True)
    build.mkdir(parents=True)
    # Send the notebook without its old outputs: Kaggle recomputes them, and saved figures make the upload too big.
    nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    source = base64.b64encode(json.dumps(nb).encode("utf-8")).decode()
    (build / f"run_{name}.py").write_text(RUNNER.format(name=name, source=source))
    meta = {
        "id": f"{USER}/{spec['slug']}",
        "title": spec["title"],
        "code_file": f"run_{name}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": spec.get("gpu", False),
        "enable_internet": spec.get("internet", False),
        "dataset_sources": spec.get("datasets", []),
        "kernel_sources": [f"{USER}/{slug}" for slug in spec.get("kernels", [])],
        "competition_sources": [],
    }
    (build / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(kaggle("kernels", "push", "-p", str(build)).strip())
    return meta["id"]


def wait(ref):
    # A new version sits in the queue for a few seconds before it starts.
    time.sleep(20)
    while True:
        status = kaggle("kernels", "status", ref).strip()
        print(time.strftime("%H:%M:%S"), status, flush=True)
        state = status.upper()
        if "COMPLETE" in state:
            return True
        if "ERROR" in state or "CANCEL" in state:
            return False
        time.sleep(30)


def collect(ref, build):
    out = build / "output"
    kaggle("kernels", "output", ref, "-p", str(out), "-o", "-q")
    # The run writes to /kaggle/working/{notebooks,data,outputs}, the same layout as this folder.
    for sub in ("notebooks", "data", "outputs"):
        if (out / sub).exists():
            shutil.copytree(out / sub, ROOT / sub, dirs_exist_ok=True)
    return out


def main():
    name = sys.argv[1]
    spec = json.loads((ROOT / "kaggle" / "kernels.json").read_text())[name]
    build = ROOT / "kaggle" / "_build" / name
    if "--no-push" in sys.argv:
        ref = f"{USER}/{spec['slug']}"
    else:
        ref = push(name, spec, build)
    ok = wait(ref)
    out = collect(ref, build)
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    print("run finished" if ok else "run FAILED", f"| {len(files)} files:", files[:40])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
