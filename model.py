import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LRScheduler
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
from math import pi as PI
import Angular_Deviation_Diffuser.util as util
import Angular_Deviation_Diffuser.loss as loss


def wrap(x):
    return torch.remainder(x + PI, 2 * PI) - PI



class LinearAnnealingLR(LRScheduler):
    def __init__(self, optimizer, num_annealing_steps, num_total_steps,global_step_func):
        self.num_annealing_steps = num_annealing_steps
        self.num_total_steps = num_total_steps
        self.global_step_func = global_step_func
        super().__init__(optimizer)

    def get_lr(self):
        global_step = self.global_step_func()
        
        if global_step <= self.num_annealing_steps:
            return [
                base_lr * global_step / self.num_annealing_steps for base_lr in self.base_lrs
            ]

            
        else:
            return [
                base_lr
                * (self.num_total_steps - global_step)
                / (self.num_total_steps - self.num_annealing_steps)
                for base_lr in self.base_lrs
            ]
            


class RandomFourierFeatures(nn.Module):
    def __init__(self):
        super().__init__()

        self.w = nn.Linear(1, 192, bias=False)
        nn.init.normal_(self.w.weight, std=2 * torch.pi)
        self.w.weight.requires_grad = False

    def forward(self, t):
        t = self.w(t.float())
        return torch.cat([torch.sin(t), torch.cos(t)], axis=-1)


class FoldingDiff(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.upscale = nn.Linear(6, 384)
        self.time_embed = RandomFourierFeatures()

        config = BertConfig(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=384 * 2,
            max_position_embeddings=147,
            hidden_dropout_prob=0.1,
            position_embedding_type="relative_key",
        )
        self.encoder = BertEncoder(config)

        self.head = nn.Sequential(
            nn.Linear(384, 384),
            nn.GELU(),
            nn.LayerNorm(384),
            nn.Linear(384, 6),
        )

        self.criterion = loss.WrappedSmoothL1Loss(beta=0.1 * torch.pi)

    def forward(self, x, t):
        x = self.upscale(x) + self.time_embed(t).unsqueeze(1)

        bert_output = self.encoder(x)
        return self.head(bert_output.last_hidden_state)

    def training_step(self, batch, batch_idx):
        x, t, eps, loss_mask = batch["x"], batch["t"], batch["eps"], batch["loss_mask"]

        out = self(x, t)
        loss = self.criterion(out * loss_mask, eps * loss_mask)

        self.log_dict({"train/loss": loss}, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        lr = self.optimizers().param_groups[0]['lr']
        self.log("lr", lr, prog_bar=True, on_step=True, on_epoch=False, logger=True)

        current_step = self.global_step
        self.log("step", current_step, prog_bar=True, on_step=True, on_epoch=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, t, eps, loss_mask = batch["x"], batch["t"], batch["eps"], batch["loss_mask"]

        out = self(x, t)
        loss = self.criterion(out * loss_mask, eps * loss_mask)

        self.log_dict({"val_loss": loss}, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        
        scheduler = LinearAnnealingLR(optimizer, num_annealing_steps=1000, num_total_steps=5000,global_step_func=self.get_global_step)
 
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def get_global_step(self):

        return self.global_step