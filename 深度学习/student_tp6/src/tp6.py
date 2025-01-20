import logging
logging.basicConfig(level=logging.INFO)

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import click


from sklearn.datasets import fetch_openml

# Changer le DATA_PATH
DATA_PATH = "/tmp/mnist"


# Ratio du jeu de train à utiliser
TRAIN_RATIO = 0.05
TEST_RATIO = 0.2
def store_grad(var):
    """Stores the gradient during backward

    For a tensor x, call `store_grad(x)`
    before `loss.backward`. The gradient will be available
    as `x.grad`

    """
    def hook(grad):
        var.grad = grad
    var.register_hook(hook)
    return var


# [[STUDENT]] Implémenter

BATCH_SIZE = 300
NUM_CLASSES = 10
ITERATIONS = 1000
DEFAULT_DIMS = [100, 100, 100]
RANDOM_SEED = 14

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Téléchargement des données
all_x,all_y = fetch_openml("mnist_784",return_X_y=True,as_frame=False, data_home=DATA_PATH)
all_x = torch.tensor(all_x).view(-1,all_x.shape[1]).float()/255.
all_y = torch.tensor(all_y.astype(int)).long()

test_length = int(TEST_RATIO*all_x.shape[0])
train_images, train_labels = all_x[:test_length].reshape(-1,28,28), all_y[:test_length]
test_images, test_labels =  all_x[test_length:].reshape(-1,28,28), all_y[test_length:]

# dimension of images (flattened)
INPUT_DIM = train_images.shape[1] * train_images.shape[2]

# -- Exo 1: Dataloader


class MnistDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)


torch.manual_seed(RANDOM_SEED)
train_images = torch.FloatTensor(train_images) / 255.
ds = MnistDataset(train_images, train_labels)
test_images = torch.FloatTensor(test_images) / 255.
test_data = MnistDataset(test_images, test_labels)


train_length = int(len(ds) * TRAIN_RATIO)
train_data, val_data = random_split(ds, (train_length, len(ds) - train_length))
logging.info("Kept %d samples out of %d for training", train_length, len(ds))


class State:
    def __init__(self, path: Path, model, optim):
        self.path = path
        self.model = model
        self.optim = optim
        self.epoch = 0
        self.iteration = 0

    @staticmethod
    def load(path: Path):
        if path.is_file():
            with path.open("rb") as fp:
                state = torch.load(fp, map_location=DEVICE)
                logging.info("Starting back from epoch %d", state.epoch)
                return state
        return State(path, None, None)

    def save(self):
        savepath_tmp = self.path.parent / ("%s.tmp" % self.path.name)
        with savepath_tmp.open("wb") as fp:
            torch.save(self, fp)
        os.rename(savepath_tmp, self.path) 


NORMALIZATIONS = {
    "identity": None,
    "batchnorm": lambda dim: nn.BatchNorm1d(dim),
    "layernorm": lambda dim: nn.LayerNorm(dim)
}

class Model(nn.Module):
    def __init__(self, in_features, out_features, dims, dropouts, normalization_str="identity"):
        super().__init__()

        layers = []
        normalization = NORMALIZATIONS[normalization_str]

        self.id = f"n{normalization_str}"
        self.trackedlayers = set()
        dim = in_features

        for newdim, p in zip(dims, dropouts):
            layers.append(nn.Linear(dim, newdim))
            dim = newdim
            self.id += f"-{dim}_{p}"

            if p > 0:
                layers.append(nn.Dropout(p))

            if normalization:
                layers.append(normalization(dim))

            self.trackedlayers.add(layers[-1])

            layers.append(nn.ReLU())

        layers.append(nn.Linear(dim, out_features))
        self.layers = nn.Sequential(*layers)
    
    def forwards(self, input):
        outputs = []
        for module in self.layers:
            input = module(input)
            if module in self.trackedlayers:
                outputs.append(store_grad(input))


        return input, outputs

    def forward(self, input):
        return self.layers(input)



def run(iterations, model, l1, l2):
    """Run a model""" 
    model_id = model.id
    if l1 >  0:
        model_id += f"-l1_{l1:.2g}"
    if l2 >  0:
        model_id += f"-l2_{l2:.2g}"
    cumloss = torch.tensor(0)
    savepath = Path(f"models/model-{model_id}.pth")
    writer = SummaryWriter(f"runs/{model_id}")

    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)


    state = State.load(savepath)
    if state.model is None:
        state.model =  model.to(DEVICE)
        state.optim = torch.optim.Adam(state.model.parameters(), lr=1e-4)

    it = 0
    model = state.model
    loss = nn.CrossEntropyLoss()
    loss_nagg = nn.CrossEntropyLoss(reduction='sum')

    def batches(loader):
        for x, y in loader:
            x = x.to(DEVICE).reshape(x.shape[0], INPUT_DIM)
            y = y.long().to(DEVICE)
            yield x, y

    for epoch in tqdm(range(state.epoch, iterations)):
        # Iterate over batches
        model.train()
        for x, y in batches(train_loader):
            state.optim.zero_grad()
            l = loss(model(x), y)

            total_loss = l

            if l1 > 0:
                l1_loss = 0
                for name, value in model.named_parameters():
                    if name.endswith(".weight"):
                        l1_loss += value.abs().sum()

                l1_loss *= l1
                total_loss += l1_loss
                writer.add_scalar('loss/l1', l1_loss, state.iteration)
                
            if l2 > 0:
                l2_loss = 0
                for name, value in model.named_parameters():
                    if name.endswith(".weight"):
                        l2_loss += (value ** 2).sum()
                l2_loss *= l2
                total_loss += l2_loss
                writer.add_scalar('loss/l2', l2_loss, state.iteration)

            total_loss.backward()
            state.optim.step()

            writer.add_scalar('loss/train', l, state.iteration)
            state.iteration += 1

            if state.iteration % 500 == 0:
                logprobs, outputs = model.forwards(x)
                with torch.no_grad():
                    probs = nn.functional.softmax(logprobs, dim=1)
                    writer.add_histogram(f'entropy', -(probs * probs.log()).sum(1), state.iteration)

                l = loss(logprobs, y)
                l.backward()
                for ix, output in enumerate(outputs):
                    writer.add_histogram(f'output/{ix}', output, state.iteration)
                    writer.add_histogram(f'grads/{ix}', output.grad, state.iteration)

                ix = 0
                for module in model.layers:
                    if isinstance(module, nn.Linear):
                        writer.add_histogram(f'linear/{ix}/weight', module.weight, state.iteration)
                        ix += 1

        # Evaluate
        model.eval()
        with torch.no_grad():
            cumloss = 0
            cumcorrect = 0
            count = 0
            for x, y in batches(test_loader):
                logprobs = model(x)
                cumloss += loss_nagg(logprobs, y)
                cumcorrect += (logprobs.argmax(1) == y).sum()
                count += x.shape[0]

            writer.add_scalar('loss/test', cumloss.item() / count, state.iteration)
            writer.add_scalar('correct/test', cumcorrect.item() / count, state.iteration)


        state.epoch = epoch + 1
        state.save()


    # Renvoie la dernière loss en test
    return cumloss.item() / count


def model(dims, dropouts, normalization="identity", l1=0, l2=0):
    return Model(INPUT_DIM, NUM_CLASSES, dims, dropouts, normalization_str=normalization), l1, l2



@click.group()
@click.option('--iterations', default=ITERATIONS, help='Number of iterations.')
@click.option('--device', default='cpu', help='Device for computation')
@click.pass_context
def cli(ctx, iterations, device): 
    global DEVICE
    ctx.obj["iterations"] = iterations
    DEVICE = torch.device(device)


@cli.command()
@click.pass_context
def vanilla(ctx):
    run(ctx.obj["iterations"], *model(DEFAULT_DIMS, [0,0,0]))

@cli.command()
@click.pass_context
def l2(ctx):
    run(ctx.obj["iterations"], *model(DEFAULT_DIMS, [0,0,0], l2=1e-3))

@cli.command()
@click.pass_context
def l1(ctx):
    run(ctx.obj["iterations"], *model(DEFAULT_DIMS, [0,0,0], l1=1e-4))

@cli.command()
@click.pass_context
def dropout(ctx):
    run(ctx.obj["iterations"], *model(DEFAULT_DIMS, [0.2,0.2,0.2]))

@cli.command()
@click.pass_context
def batchnorm(ctx):
    run(ctx.obj["iterations"], *model(DEFAULT_DIMS, [0,0,0], 'batchnorm'))


if __name__ == '__main__':
    cli(obj={})

# [[/STUDENT]]
