import torch
from torch.nn.utils         import clip_grad_norm_
from src.loss               import _loss

def _train(
        model, 
        constraint_func, 
        dataloader, 
        optim, 
        args, 
        lmda
):
    model.train()

    train_task_loss = 0.
    train_constraint_loss = 0.
    n_train = 0
    
    for x, y, s in dataloader:
        batch_size = x.size(0)
        n_train += batch_size

        optim.zero_grad()

        task_loss, constraint_loss = _loss(
            x, y, s, 
            model, constraint_func, 
            args
        )
        
        loss = task_loss + lmda * constraint_loss

        loss.backward()
        if args.clip_grad:
            clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optim.step()

        train_task_loss += task_loss.item() * batch_size
        train_constraint_loss += constraint_loss.item() * batch_size
    
    train_task_loss /= n_train
    train_constraint_loss /= n_train

    return train_task_loss, train_constraint_loss