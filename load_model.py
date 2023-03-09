



import torch
from trainer3a import * 
from task3_model1 import * 
from load_my_custom import * 
from create_plots import * 

epochs = 4
batch_size = 32
learning_rate = 1e-2 # Should be 5e-5 for LeNet
early_stop_count = 3

num_classes = 9
model = Model1(3, 9) 

data = get_data(1, transform)
dataloaders = create_dataloaders(data, batch_size)

trainer = Trainer3(
    batch_size,
    learning_rate,
    early_stop_count,
    epochs,
    model,
    dataloaders
)

trainer.load_best_model()

create_plots(trainer, "test")