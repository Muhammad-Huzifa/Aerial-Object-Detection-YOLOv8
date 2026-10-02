"""Check dataset arguments and checkpoint export without training a model."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from aerial_detection import paths, training


class TrainingTests(unittest.TestCase):
    def test_directory_and_yaml_path_resolve_to_the_same_configuration(self):
        with TemporaryDirectory() as directory:
            data = Path(directory) / "data.yaml"
            data.write_text("names: [car]\n")
            self.assertEqual(training.resolve_data_yaml(directory), data.resolve())
            self.assertEqual(training.resolve_data_yaml(data), data.resolve())

    def test_missing_data_has_a_clear_error(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Dataset configuration"):
                training.resolve_data_yaml(Path(directory) / "missing.yaml")

    def test_export_uses_actual_trainer_directory(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "custom-run" / "weights" / "best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"test checkpoint")
            model = SimpleNamespace(trainer=SimpleNamespace(best=checkpoint))
            destination = Path(directory) / "models" / "best.pt"
            exported = training.export_best_checkpoint(model, destination)
            self.assertEqual(exported, destination)
            self.assertEqual(destination.read_bytes(), checkpoint.read_bytes())

    def test_export_rejects_missing_checkpoint(self):
        with TemporaryDirectory() as directory:
            model = SimpleNamespace(trainer=SimpleNamespace(best=Path(directory) / "missing.pt"))
            with self.assertRaisesRegex(FileNotFoundError, "best checkpoint"):
                training.export_best_checkpoint(model, Path(directory) / "export.pt")

    def test_export_does_not_copy_a_checkpoint_over_itself(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            checkpoint.write_bytes(b"test checkpoint")
            model = SimpleNamespace(trainer=SimpleNamespace(best=checkpoint))
            self.assertEqual(training.export_best_checkpoint(model, checkpoint), checkpoint)

    def test_download_requires_an_environment_key(self):
        with patch.dict(os.environ, {"ROBOFLOW_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "ROBOFLOW_API_KEY"):
                training.download_dataset("workspace", "project")

    def test_download_uses_yolov8_format_and_configured_version(self):
        with TemporaryDirectory() as directory:
            data = Path(directory) / "data.yaml"
            data.write_text("names: [car]\n")
            sdk = MagicMock()
            project = sdk.Roboflow.return_value.workspace.return_value.project.return_value
            project.version.return_value.download.return_value.location = directory
            with patch.dict(os.environ, {"ROBOFLOW_API_KEY": "test-only-key"}), \
                 patch.dict("sys.modules", {"roboflow": sdk}), \
                 patch.object(training, "DATA_DIR", Path(directory) / "downloads"):
                result = training.download_dataset("workspace", "project", 3)
            self.assertEqual(result, data.resolve())
            project.version.assert_called_once_with(3)
            self.assertEqual(project.version.return_value.download.call_args.args[0], "yolov8")

    def test_training_passes_configuration_and_returns_the_model(self):
        with TemporaryDirectory() as directory:
            data = Path(directory) / "data.yaml"
            data.write_text("names: [car]\n")
            torch = MagicMock()
            torch.cuda.is_available.return_value = False
            ultralytics = MagicMock()
            with patch.dict("sys.modules", {"torch": torch, "ultralytics": ultralytics}):
                model = training.train_model(data, project=Path(directory) / "runs", epochs=2)
            self.assertIs(model, ultralytics.YOLO.return_value)
            arguments = model.train.call_args.kwargs
            self.assertEqual(arguments["data"], str(data.resolve()))
            self.assertEqual(arguments["epochs"], 2)
            self.assertEqual(arguments["device"], "cpu")
            self.assertFalse(arguments["exist_ok"])

    def test_model_path_honors_environment_and_explicit_override(self):
        with patch.dict(os.environ, {"AERIAL_MODEL_PATH": "models/custom.pt"}):
            self.assertEqual(paths.model_path(), paths.PROJECT_ROOT / "models/custom.pt")
            self.assertEqual(paths.model_path("models/other.pt"), paths.PROJECT_ROOT / "models/other.pt")


if __name__ == "__main__":
    unittest.main()
