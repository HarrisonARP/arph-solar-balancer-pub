"""Execute the shipped notebooks, because nothing else does.

The notebook cross-checks its hand-built energy breakdown against
``model.calculate_plant_energy``. That assert is the whole point of it, and it only
fires when the notebook actually runs - twice a model change has silently broken it
while every other test stayed green.

Jupyter is not in requirements.txt and the user guide promises the suite needs no extra
packages, so this skips rather than fails when nbclient is absent.
"""
from pathlib import Path
import unittest

try:  # pragma: no cover - import guard, not logic
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError
    JUPYTER = True
except ImportError:  # pragma: no cover
    JUPYTER = False

ROOT = Path(__file__).resolve().parents[1]
# The walkthrough dispatches a full ten-year evaluation, so this takes minutes. That is
# the price of knowing the worked example still runs against the current model.
NOTEBOOKS = (ROOT / "sab_notebook.ipynb",)


@unittest.skipUnless(JUPYTER, "nbformat/nbclient not installed")
class NotebookTests(unittest.TestCase):
    def test_notebooks_run_and_match_the_model(self):
        """Every cell executes, which means each notebook's E_req assert passed."""
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                self.assertTrue(path.exists(), f"{path.name} is missing")
                notebook = nbformat.read(path, as_version=4)
                # Run from the repository root so imports resolve the way they do for
                # a reader who opened the file there.
                client = NotebookClient(
                    notebook, timeout=1800, kernel_name="python3",
                    resources={"metadata": {"path": str(path.parent)}},
                )
                try:
                    client.execute()
                except CellExecutionError as error:  # pragma: no cover - failure path
                    self.fail(f"{path.name} failed to execute:\n{error}")

                printed = "".join(
                    "".join(output.get("text", ""))
                    for cell in notebook.cells
                    for output in cell.get("outputs", [])
                    if output.get("output_type") == "stream"
                )
                self.assertIn(
                    "E_req", printed,
                    f"{path.name} no longer reports E_req; its cross-check may be gone")

    def test_notebooks_ship_without_saved_outputs(self):
        """Outputs are cleared, so each file is a source document, not a record."""
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                notebook = nbformat.read(path, as_version=4)
                with_outputs = [
                    index for index, cell in enumerate(notebook.cells) if cell.get("outputs")
                ]
                self.assertEqual(
                    with_outputs, [],
                    f"{path.name} cells {with_outputs} carry saved outputs; "
                    "clear them before committing")
