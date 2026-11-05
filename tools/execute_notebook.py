"""Execute the distributed notebook, saving completed cells even on failure."""
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / "tmp/ipython"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / "tmp/jupyter"))
os.environ.setdefault("QF_TEXT_ONLY", "1")
os.environ["QF_SD_REVISION"] = (ROOT / "tmp/sd_revision.txt").read_text().strip()
kernel_dir = ROOT / "tmp/kernels/qf-local"
kernel_dir.mkdir(parents=True, exist_ok=True)
import json, sys
(kernel_dir / "kernel.json").write_text(json.dumps(dict(
    argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    display_name="QF local verification", language="python")))
ksm = KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)])
notebook = nbformat.read(ROOT / "QF_Attack_Colab.ipynb", as_version=4)
client = NotebookClient(notebook, timeout=3600, kernel_name="qf-local",
                        resources={"metadata": {"path": str(ROOT)}})
client.on_cell_start = lambda cell, cell_index, **kwargs: print(
    f"Starting cell {cell_index}: {cell.get('id')}", flush=True)
from jupyter_client import KernelManager
client.km = KernelManager(kernel_name="qf-local", kernel_spec_manager=ksm)
try:
    client.execute()
    print("All notebook cells executed successfully", flush=True)
finally:
    nbformat.write(notebook, ROOT / "QF_Attack_Executed.ipynb")
    print("Saved QF_Attack_Executed.ipynb", flush=True)
    if client.km is not None and client.km.has_kernel:
        client.km.shutdown_kernel(now=True)
