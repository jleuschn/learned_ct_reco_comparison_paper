from tqdm import tqdm
import torch
from ista_unet.utils import seed_everything
import torch.nn as nn

def fit_model_with_loaders(model, optimizer, scheduler, num_epochs, criterion, loaders, device, max_grad_norm = 1, seed = 0):

    train_loader = loaders['train']    
    len_train_loader = len(train_loader) 
    seed_everything(seed = seed)
    
    print('start training')
    for epoch in tqdm(range(num_epochs) ) :
        loss = 0
        for i, (x, d) in enumerate(train_loader):
            x, d = x.cuda(device), d.cuda(device)
                    
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions
            outputs = model(x)
            
            # compute training reconstruction loss
            train_loss = criterion(outputs, d) 
            
            # compute accumulated gradients
            train_loss.backward()
            
            # clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = max_grad_norm)            

            # perform parameter update based on current gradients
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss +=  float(train_loss) 

        # compute the epoch training loss
        loss = float(loss) / len_train_loader   
        
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, num_epochs, loss))
        
        # update the step-size
        scheduler.step() 

    return model



def fit_model_with_loaders_verbose_iter(model, optimizer, scheduler, num_epochs, criterion, loaders, device, verbose_every = 100, max_grad_norm = 1, seed = 0):

    train_loader = loaders['train']    
    len_train_loader = len(train_loader) 
    seed_everything(seed = seed)
    
    print('start training')
    for epoch in tqdm(range(num_epochs) ) :
        loss = 0
        for i, (x, d) in enumerate(train_loader):
            x, d = x.cuda(device), d.cuda(device)
                    
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions
            outputs = model(x)
            
            # compute training reconstruction loss
            train_loss = criterion(outputs, d)
            
            # compute accumulated gradients
            train_loss.backward()
            
            # clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = max_grad_norm)            

            # perform parameter update based on current gradients
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss +=  float(train_loss) 
            
            if i % verbose_every == 0:
                print("iter : {}/{}, loss = {:.6f}".format(epoch * len(loaders['train']) + i, len(loaders['train']) * num_epochs, float(train_loss)))
         
        # compute the epoch training loss
        loss = float(loss) / len_train_loader   
        
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, num_epochs, loss))
        
        # update the step-size
        scheduler.step() 

    return model
