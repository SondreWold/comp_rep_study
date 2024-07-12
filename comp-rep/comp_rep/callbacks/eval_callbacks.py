"""
Callbacks for evaluation.
"""

from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader

from comp_rep.eval.decoding import GreedySearch
from comp_rep.eval.evaluator import evaluate_generation


class TestGenerationCallback(Callback):
    def __init__(
        self,
        frequency: int,
        searcher: GreedySearch,
        test_loader: DataLoader,
        device: str,
    ):
        self.frequency = frequency
        self.searcher = searcher
        self.test_loader = test_loader
        self.device = device

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Callback function to evaluate the model's accuracy.

        Args:
            trainer (Trainer): The PyTorch Lightning trainer object.
            pl_module (LightningModule): The PyTorch Lightning module.

        Returns:
            None
        """
        epoch = trainer.current_epoch

        if epoch % self.frequency == 0:
            acc = evaluate_generation(
                model=pl_module.model,
                searcher=self.searcher,
                test_loader=self.test_loader,
                device=self.device,
            )
            pl_module.log("val_accuracy", acc)
