import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from trainer import Trainer, compute_loss_and_accuracy

def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    #her er problemet som rammes av cpu/gpu feil
    utils.plot_loss(trainer.validation_history["accuracy"].cpu(), label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}.png"))
    plt.show()
