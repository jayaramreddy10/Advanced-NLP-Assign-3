import os
import copy
import numpy as np
import torch
import einops
import pdb
import os
from timer import Timer
from arrays import batch_to_device, to_np, apply_dict, to_device
from torch import nn
import wandb
import time
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.functional import to_map_style_dataset
from torch.nn.functional import pad

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cycle(dl):
    while True:
        for data in dl:
            yield data

# Custom collate function
def collate_fn_tmp(batch):
    # Pad sequences in the batch to have the same length
    padded_batch = [torch.tensor(sentence) for sentence in batch]
    padded_batch = pad_sequence(padded_batch, batch_first=True, padding_value='<pad>')
    return padded_batch

# Custom collate function
fixed_length = 50
def collate_fn(batch):
    padded_batch = batch
    # Pad sequences in the batch to have the same length
    for i, sentence in enumerate(batch):
        words = sentence.split(" ")
        words = [string for string in words if string]
        if len(words) >= fixed_length:
            padded_batch[i] = words[:fixed_length]  # Truncate if longer
            padded_batch[i].insert(0, '<sos>')
            padded_batch[i].append('<eos>')
        else:
            num_padding = fixed_length - len(words) 
            padded_batch[i] = words
            padded_batch[i].insert(0, '<sos>')
            padded_batch[i].append('<eos>')
            padded_batch[i] = padded_batch[i] + ['<pad>'] * num_padding
            
    # padded_batch = [torch.tensor(sentence.split(" ")) for sentence in batch]
    # padded_batch = pad_sequence(padded_batch, batch_first=True, padding_value='<pad>')
    return padded_batch

SOS_IDX = 0
EOS_IDX = 1
PAD_IDX = 2
MAX_PADDING = 20

def generate_batch(data_batch):

  eng_batch, fr_batch = [], []

  # for each sentence
  for (eng_item, fr_item) in data_batch:
    # add <bos> and <eos> indices before and after the sentence
    eng_temp = torch.cat([torch.tensor([SOS_IDX]), eng_item, torch.tensor([EOS_IDX])], dim=0).to(device)
    fr_temp = torch.cat([torch.tensor([SOS_IDX]), fr_item, torch.tensor([EOS_IDX])], dim=0).to(device)

    # add padding
    eng_batch.append(pad(eng_temp,(0, # dimension to pad
                            MAX_PADDING - len(eng_temp), # amount of padding to add
                          ),value=PAD_IDX,))
    
    # add padding
    fr_batch.append(pad(fr_temp,(0, # dimension to pad
                            MAX_PADDING - len(fr_temp), # amount of padding to add
                          ),
                          value=PAD_IDX,))
    
  return torch.stack(eng_batch), torch.stack(fr_batch)

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer_transformer_jayaram(object):  
    def __init__(
        self,
        model,
        train_dataset, 
        val_dataset, 
        is_LSTM = False, 
        device = 'cuda:0',
        ema_decay=0.995,
        train_batch_size=64,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=40000,
        save_parallel=False,
        # results_folder='/home/jayaram/research/research_tracks/table_top_rearragement/test_diffusion_planning/logs/maze2d-test/diffusion',
        results_folder='/home/jayaram/research/NLP_course_papers_and_content/Assignments/Assign_3/logs/Transformer',
        n_reference=50,
        n_samples=10,
        bucket=None,
    ):
        super().__init__()
        self.model = model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)

        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        # self.train_dataloader = cycle(torch.utils.data.DataLoader(
        #     self.train_dataset, batch_size=train_batch_size, collate_fn = collate_fn, num_workers=1, shuffle=True, pin_memory=True
        # ))
        self.train_dataloader = cycle(torch.utils.data.DataLoader(to_map_style_dataset(self.train_dataset), batch_size=train_batch_size,
                        shuffle=True, drop_last=True, collate_fn=generate_batch))

        self.val_dataloader = cycle(torch.utils.data.DataLoader(to_map_style_dataset(self.val_dataset), batch_size=train_batch_size,
                        shuffle=True, drop_last=True, collate_fn=generate_batch))
        
        # self.dataloader_vis = cycle(torch.utils.data.DataLoader(
        #     self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        # ))
        self.optimizer = torch.optim.Adam(model.parameters(), lr=train_lr)

        if(is_LSTM):
            self.logdir = '/home/jayaram/research/NLP_course_papers_and_content/Assignments/Assign_3/logs/Transformer'
        else: 
            self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.n_samples = n_samples
        self.criterion = nn.CrossEntropyLoss() #nn.CrossEntropyLoss(ignore_index = 2)  # Mean Squared Error loss
        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())


    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)


    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train_Transformer(self, device, epoch_no, n_train_steps):
        isExist = os.path.exists(self.logdir)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.logdir)
            print("The new directory is created!")  
            
        timer = Timer()
        for step in range(n_train_steps):
            loss = 0.0
            for i in range(self.gradient_accumulate_every):
                batch = next(self.train_dataloader)   #add collate fn here (not necessary, handled in __get_item itself)
                src, trg = batch

                src = to_device(src)
                trg = to_device(trg)
                
                logits = self.model(src, trg[:,:-1])   

                # expected output
                expected_output = trg[:,1:]

                # calculate the loss
                loss = self.criterion(logits.contiguous().view(-1, logits.shape[-1]), 
                      expected_output.contiguous().view(-1))
                
                loss = loss / self.gradient_accumulate_every
                
                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:
                #         print(f'Parameter: {name}, Gradient norm: {param.grad.norm()}')

                loss.backward()  

            #scale back loss
            loss = loss * self.gradient_accumulate_every
            wandb.log({'train loss': loss, 'epoch': epoch_no, 'step no': step}) #, 'batch': t})
           
            # train_perplexity = compute_perplexity()
            # wandb.log({'train_perplexity': train_perplexity, 'epoch': epoch_no, 'step no': step}) #, 'batch': t})
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.log_freq == 0:
                print(f'{self.step}: {loss:8.4f}  | t: {timer():8.4f}')
            self.step += 1

        label = epoch_no
        self.save(label)

        # report validation loss/scores
        validation_loss = 0.0    #for entire val dataset in current epoch

        # Set your model to evaluation mode
        self.model.eval()
        # Iterate through the validation dataset
        # print('val dataloader len: {}'.format(len(self.val_dataloader)))

        with torch.no_grad():
            for val_batch in self.val_dataloader:
                # time_s = time.time()

                # val_batch = next(self.val_dataloader)   #add collate fn here (not necessary, handled in __get_item itself)

                src, trg = val_batch

                src = to_device(src)
                trg = to_device(trg)
                
                logits = self.model(src, trg[:,:-1])   

                # expected output
                expected_output = trg[:,1:]

                # calculate the loss
                val_loss = self.criterion(logits.contiguous().view(-1, logits.shape[-1]), 
                      expected_output.contiguous().view(-1))
                validation_loss += val_loss    

                # time_e = time.time()
                # print('one batch val time: {}'.format(time_e - time_s))

        average_validation_loss = validation_loss / len(self.val_dataloader)
        # log results to text file
        print(f'Validation Loss: {average_validation_loss:.4f}, Epoch: {epoch_no}')
        wandb.log({'val loss ': average_validation_loss, 'epoch': epoch_no}) #, 'batch': t})

        # Set your model back to training mode
        self.model.train()

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        isExist = os.path.exists(self.logdir)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.logdir)
            print("The new directory is created!")        

        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])



