import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


from models.Seq2SeqTranslator import *
from data.Multi30k import getMulti30kDataloadersAndVocabs

import wandb

import matplotlib.pyplot as plt
import datetime
import random
import string
import wandb
from tqdm import tqdm

use_cuda_if_avail = True
if use_cuda_if_avail and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

config = {
    "bs":128,   # batch size
    "lr":0.001, # learning rate
    "l2reg":0.0000001, # weight decay
    "max_epoch":50,
    "embed_dim":128,
    "enc_dim":256,
    "dec_dim":256,
    "kq_dim":64,
    "attn": "mean", #Options are none, mean, dotproduct
    "dropout":0.5
}



def main():
    
    train_loader, val_loader, test_loader, vocabs = getMulti30kDataloadersAndVocabs(config["bs"])

    de_to_ge_model = Seq2Seq(len(vocabs["de"]), len(vocabs["en"]), config["embed_dim"], 
                    config["enc_dim"], config["dec_dim"], config["kq_dim"], config["attn"], config["dropout"])
    print(de_to_ge_model)

    torch.compile(de_to_ge_model)

    train(de_to_ge_model, train_loader, val_loader, vocabs)



def train(model, train_loader, val_loader, vocabs):

  # Log our exact model architecture string
  config["arch"] = str(model)
  run_name = generateRunName()

  # Startup wandb logging
  wandb.login()
  wandb.init(project="Multi30K CS499 A7", name=run_name, config=config)

  # Move model to the GPU
  model.to(device)

  # Set up optimizer and our learning rate schedulers
  optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["l2reg"])
  warmup_epochs = config["max_epoch"]//10
  linear = LinearLR(optimizer, start_factor=0.25, total_iters=warmup_epochs)
  cosine = CosineAnnealingLR(optimizer, T_max = config["max_epoch"]-warmup_epochs)
  scheduler = SequentialLR(optimizer, schedulers=[linear, cosine], milestones=[warmup_epochs])

  # Loss
  criterion = nn.CrossEntropyLoss(ignore_index=0)

  # Main training loop with progress bar
  iteration = 0
  pbar = tqdm(total=config["max_epoch"]*len(train_loader), desc="Training Iterations", unit="batch")
  for epoch in range(config["max_epoch"]):
    model.train()

    # Log LR
    wandb.log({"LR/lr": scheduler.get_last_lr()[0]}, step=iteration)

    for x, y, src_lens , _ in train_loader:
      x = x.to(device)
      y = y.to(device)
      src_lens = src_lens.to(device)

      out = model(x, y, src_lens)
      
      loss = criterion(out.permute(0,2,1), y)

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      
      nonpad = (y != 0).to(dtype=float).sum().item()
      acc = (torch.argmax(out, dim=2)==y).to(dtype=float).sum() / nonpad

      wandb.log({"Loss/train": loss.item(), "Acc/train": acc.item()}, step=iteration)
      pbar.update(1)
      iteration+=1

    val_loss, val_acc = evaluate(model, val_loader)
    wandb.log({"Loss/val": val_loss, "Acc/val": val_acc}, step=iteration)
    
    if epoch % 5 == 0:
      figs = generateAttentionTranslationPlots(model, val_loader, vocabs)
      for i,f in enumerate(figs):
        wandb.log({"Viz/attn"+str(i):f}, step=iteration)
        plt.close(f)

    # Adjust LR
    scheduler.step()

  wandb.finish()
  pbar.close()



def evaluate(model, loader):
  model.eval()

  running_loss = 0
  running_acc = 0
  criterion = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=0)

  nonpad = 0
  for x,y, src_lens, _ in loader:

    x = x.to(device)
    y = y.to(device)
    src_lens = src_lens.to(device)

    out = model(x, y, src_lens)
    loss = criterion(out.permute(0,2,1), y)
    acc = (torch.argmax(out, dim=2)==y).to(dtype=float).sum()
    

    nonpad += (y != 0).to(dtype=float).sum().item()
    
    running_loss += loss.item()
    running_acc += acc.item()

  return running_loss/nonpad, running_acc/nonpad



def generateAttentionTranslationPlots(model, val_loader, vocabs, max_len=25):
  model.eval()

  sos_id = vocabs["en"].word2idx["<SOS>"]

  # Get a batch
  x, y ,src_lens, _ = next(iter(val_loader))

  out, attn = model.translate(x.to(device),src_lens.to(device), sos_id, max_len)

  figs = []
  for i in range(min(8,x.shape[0])):
    translation = vocabs["en"].idx2text(out[i,:].detach().cpu().numpy())
    translation = [t.strip() for t in translation]
    for t, w in enumerate(translation):
       if w == '<EOS>':
          translation = translation[:t]
          break
       
    attn_mat = attn[i,:t,:].detach().cpu().numpy()

    source = vocabs["de"].idx2text(x[i,:].detach().cpu().numpy())
    source = [s.strip() for s in source]
    
    actual_translation = vocabs["en"].idx2text(y[i,:].detach().cpu().numpy())
    for t, w in enumerate(actual_translation):
       if w == '<EOS>':
          actual_translation = actual_translation[1:t]
          break
    
  
        
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    

    ax.matshow(attn_mat, cmap='Greys_r', vmin=0, vmax=1)
    

    ax.tick_params(labelsize=15)

    x_ticks = source
    y_ticks = translation

  
    ax.set_xticks(torch.arange(len(x_ticks)), x_ticks, rotation='vertical')
    ax.set_yticks(torch.arange(len(y_ticks)), y_ticks)
    ax.tick_params(axis='both', labelsize=10)

    ax.set_title(" ".join(actual_translation), y=-0.01, pad=-18)
    fig.tight_layout()


    figs.append(fig)

    

  return figs

        


def generateRunName():
  random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
  now = datetime.datetime.now()
  run_name = ""+random_string+"_Multi30k"
  return run_name

if __name__== "__main__":
    main()