import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,random_split,TensorDataset
from pathlib import Path
import time
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

BATCH_SIZE = 311
TRAIN_RATIO = 0.8
TEST_RATIO = 0.1
LOG_PATH = "/tmp/runs/lightning_logs"

# Changer le DATA_PATH
DATA_PATH = "/tmp/mnist"

from sklearn.datasets import fetch_openml

class Lit2Layer(pl.LightningModule):
    def __init__(self,dim_in,l,dim_out,learning_rate=1e-3):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(dim_in,l),nn.ReLU(),nn.Linear(l,dim_out))
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss()
        self.name = "exemple-lightning"
        self.valid_outputs = []
        self.training_outputs = []

    def forward(self,x):
        """ Définit le comportement forward du module"""
        x = self.model(x) 
        return x
         
    def configure_optimizers(self):
        """ Définit l'optimiseur """
        optimizer = torch.optim.Adam(self.parameters(),lr=self.learning_rate)
        return optimizer
    
    def training_step(self,batch,batch_idx):
        """ une étape d'apprentissage
        doit retourner soit un scalaire (la loss), 
        soit un dictionnaire qui contient au moins la clé 'loss'"""
        x, y = batch
        yhat= self(x) ## equivalent à self.model(x)
        loss = self.loss(yhat,y)
        acc = (yhat.argmax(1)==y).sum()
        logs = {"loss":loss,"accuracy":acc,"nb":len(x)}
        self.log("accuracy",acc/len(x),on_step=False,on_epoch=True)
        self.valid_outputs.append({"loss":loss,"accuracy":acc,"nb":len(x)})
        return logs
    
    def validation_step(self,batch,batch_idx):
        """ une étape de validation
        doit retourner un dictionnaire"""
        x, y = batch
        yhat = self(x)
        loss = self.loss(yhat,y)
        acc = (yhat.argmax(1)==y).sum()
        logs = {"loss":loss,"accuracy":acc,"nb":len(x)}
        self.log("val_accuracy", acc/len(x),on_step=False,on_epoch=True)
        self.valid_outputs.append({"loss":loss,"accuracy":acc,"nb":len(x)})
        return logs
    
    def test_step(self,batch,batch_idx):
        """ une étape de test """
        x, y = batch
        yhat = self(x)
        loss = self.loss(yhat,y)
        acc = (yhat.argmax(1)==y).sum()
        logs = {"loss":loss,"accuracy":acc,"nb":len(x)}
        return logs
    
    def log_x_end(self,outputs,phase):
        total_acc = sum([o['accuracy'] for o in outputs])
        total_nb = sum([o['nb'] for o in outputs])
        total_loss = sum([o['loss'] for o in outputs])/len(outputs)
        total_acc = total_acc/total_nb
        self.log_dict({f"loss/{phase}":total_loss,f"acc/{phase}":total_acc})
        #self.logger.experiment.add_scalar(f'loss/{phase}',total_loss,self.current_epoch)
        #self.logger.experiment.add_scalar(f'acc/{phase}',total_acc,self.current_epoch)
    
    def on_training_epoch_end(self):
        """ hook optionel, si on a besoin de faire quelque chose apres une époque d'apprentissage.
        Par exemple ici calculer des valeurs à logger"""
        self.log_x_end(self.training_outputs,'train')
        self.training_outputs.clear()
        # Le logger de tensorboard est accessible directement avec self.logger.experiment.add_XXX
    def on_validation_epoch_end(self):
        """ hook optionel, si on a besoin de faire quelque chose apres une époque de validation."""
        self.log_x_end(self.valid_outputs,'valid')
        self.valid_outputs.clear()
        
    def on_test_epoch_end(self):
        pass
        



class LitMnistData(pl.LightningDataModule):
        
    def __init__(self,batch_size=BATCH_SIZE,train_ratio=TRAIN_RATIO,test_ratio=TEST_RATIO):
        super().__init__()
        self.dim_in = None
        self.dim_out = None
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

    def prepare_data(self):
        ### Do not use "self" here.
        # S'il faut charger d'abord les données ou autres.
        pass
            
    def setup(self,stage=None):
        x,y = fetch_openml("mnist_784",return_X_y=True,as_frame=False, data_home=DATA_PATH)
        x = torch.tensor(x).view(-1,x.shape[1]).float()/255.
        y = torch.tensor(y.astype(int)).long()
        train_length = int(x.shape[0]*self.train_ratio)
        test_length = int(x.shape[0]*self.test_ratio)
        val_length = x.shape[0]-train_length-test_length
        if stage =="fit" or None:
            # Si on est en phase d'apprentissage ou val
            x,y  = x[:x.shape[0]-test_length], y[:x.shape[0]-test_length]
            self.dim_in = x.shape[1]
            self.dim_out = len(set(y))
            ds_train = TensorDataset(x,y)
            self.mnist_train, self.mnist_val, = random_split(ds_train,[train_length,x.shape[0]-train_length])
        if stage == "test":
            # en phase de test
            x,y = x[x.shape[0]-test_length:],y[x.shape[0]-test_length:]
            self.mnist_test= TensorDataset(x,y)
            
    def train_dataloader(self):
        return DataLoader(self.mnist_train,batch_size=self.batch_size)
    def val_dataloader(self):
        return DataLoader(self.mnist_val,batch_size=self.batch_size)
    def test_dataloader(self):
        return DataLoader(self.mnist_test,batch_size=self.batch_size)
    

data = LitMnistData()

# Doivent etre appelées manuellement ici, car on a besoin de dim_in. Sinon, l'appel est automatique
data.prepare_data()
data.setup(stage="fit")

model = Lit2Layer(data.dim_in,10,data.dim_out,learning_rate=1e-3)

logger = TensorBoardLogger(save_dir=LOG_PATH,name=model.name,version=time.asctime(),default_hp_metric=False)

trainer = pl.Trainer(default_root_dir=LOG_PATH,logger=logger,max_epochs=100)
trainer.fit(model,datamodule=data)
trainer.test(model,datamodule=data)
