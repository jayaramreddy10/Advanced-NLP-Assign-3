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

class Trainer_jayaram(object):  
    def __init__(
        self,
        model,
        train_dataset, 
        val_dataset, 
        is_LSTM = False, 
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
        results_folder='/home/jayaram/research/NLP_course_papers_and_content/Assignments/Assign_2/logs/ELMo',
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
        self.train_dataloader = cycle(torch.utils.data.DataLoader(
            self.train_dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        )
        # self.dataloader_vis = cycle(torch.utils.data.DataLoader(
        #     self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        # ))
        self.optimizer = torch.optim.Adam(model.parameters(), lr=train_lr)

        if(is_LSTM):
            self.logdir = '/home/jayaram/research/NLP_course_papers_and_content/Assignments/Assign_2/logs/ELMo'
        else: 
            self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.n_samples = n_samples
        self.criterion = nn.CrossEntropyLoss()  # Mean Squared Error loss
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

    def train_ELMo(self, device, epoch_no, n_train_steps):
        isExist = os.path.exists(self.logdir)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.logdir)
            print("The new directory is created!")  
            
        timer = Timer()
        for step in range(n_train_steps):
            fwd_loss = 0.0
            rev_loss = 0.0
            for i in range(self.gradient_accumulate_every):
                batch_fwd, batch_rev = next(self.train_dataloader)   #add collate fn here (not necessary, handled in __get_item itself)

                # NNLM: batch[0].size: (B, window_size*embed_size)
                # LSTM: batch[0].size: (B, window_size, embed_size), we need to reshape this to ( window_size, B, embed_size) to pass it to LSTM
                batch_fwd[0] = einops.rearrange(batch_fwd[0], 'b w e -> w b e')   #ex: 52-w, 256-b, 300-e
                batch_rev[0] = einops.rearrange(batch_rev[0], 'b w e -> w b e')   #ex: 52-w, 256-b, 300-e

                # NNLM: batch[1].size: (B, )
                # LSTM: batch[1].size: (B, window_size/seq_len)
                for k, el in enumerate(batch_fwd):
                    batch_fwd[k] = to_device(batch_fwd[k])

                for k, el in enumerate(batch_rev):
                    batch_rev[k] = to_device(batch_rev[k])
                
                (_, fwd_outputs), (_, rev_outputs), _ = self.model(batch_fwd[0], batch_rev[0])   #logits in this case (seq_len, B, vocab_size) 
                # outputs = einops.rearrange(outputs, 'w b e -> b w e')   #(B, seq_len, vocab_size)
                fwd_outputs = fwd_outputs.view(-1, fwd_outputs.size(-1))      
                rev_outputs = rev_outputs.view(-1, rev_outputs.size(-1))       

                # _, label_predictions = torch.max(outputs, 1)
                fwd_targets = batch_fwd[1].T
                fwd_targets = fwd_targets.to(torch.long)   #(seq_len, B = 512)
                fwd_targets = fwd_targets.reshape(-1)

                rev_targets = batch_rev[1].T
                rev_targets = rev_targets.to(torch.long)   #(seq_len, B = 512)
                rev_targets = rev_targets.reshape(-1)

                # compute fwd and rev losses and backpropogate to train fwd_LSTM and rev_LSTM seperately.
                fwd_loss = self.criterion(fwd_outputs, fwd_targets)
                fwd_loss = fwd_loss / self.gradient_accumulate_every

                rev_loss = self.criterion(rev_outputs, rev_targets)
                rev_loss = rev_loss / self.gradient_accumulate_every
                
                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:
                #         print(f'Parameter: {name}, Gradient norm: {param.grad.norm()}')

                fwd_loss.backward()   #will train 2 layers of fwd LSTM layers
                rev_loss.backward()   #will train 2 layers of rev LSTM layers

            #scale back loss
            fwd_loss = fwd_loss * self.gradient_accumulate_every
            wandb.log({'train loss fwd_LSTM': fwd_loss, 'epoch': epoch_no, 'step no': step}) #, 'batch': t})
           
            # train_perplexity = compute_perplexity()
            # wandb.log({'train_perplexity': train_perplexity, 'epoch': epoch_no, 'step no': step}) #, 'batch': t})
            
            #scale back loss
            rev_loss = rev_loss * self.gradient_accumulate_every
            wandb.log({'train loss rev_LSTM': rev_loss, 'epoch': epoch_no, 'step no': step}) #, 'batch': t})

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.log_freq == 0:
                print(f'{self.step}: {fwd_loss:8.4f}  | t: {timer():8.4f}')
                print(f'{self.step}: {rev_loss:8.4f}  | t: {timer():8.4f}')
            self.step += 1

        label = epoch_no
        self.save(label)

        # report validation loss/scores
        fwd_validation_loss = 0.0    #for entire val dataset in current epoch
        rev_validation_loss = 0.0    #for entire val dataset in current epoch

        # Set your model to evaluation mode
        self.model.eval()
        # Iterate through the validation dataset
        # print('val dataloader len: {}'.format(len(self.val_dataloader)))

        with torch.no_grad():
            for val_batch_fwd,  val_batch_rev in self.val_dataloader:
                # time_s = time.time()

                # NNLM: batch[0].size: (B, window_size*embed_size)
                # LSTM: batch[0].size: (B, window_size, embed_size), we need to reshape this to ( window_size, B, embed_size) to pass it to LSTM
                val_batch_fwd[0] = einops.rearrange(val_batch_fwd[0], 'b w e -> w b e')   #ex: 52-w, 256-b, 300-e
                val_batch_rev[0] = einops.rearrange(val_batch_rev[0], 'b w e -> w b e')   #ex: 52-w, 256-b, 300-e

                # NNLM: batch[1].size: (B, )
                # LSTM: batch[1].size: (B, window_size/seq_len)
                for k, el in enumerate(val_batch_fwd):
                    val_batch_fwd[k] = to_device(val_batch_fwd[k])

                for k, el in enumerate(val_batch_rev):
                    val_batch_rev[k] = to_device(val_batch_rev[k])
                
                (_, val_fwd_outputs), (_, val_rev_outputs), _ = self.model(val_batch_fwd[0], val_batch_rev[0])   #logits in this case (seq_len, B, vocab_size) 
                # outputs = einops.rearrange(outputs, 'w b e -> b w e')   #(B, seq_len, vocab_size)
                val_fwd_outputs = val_fwd_outputs.view(-1, val_fwd_outputs.size(-1))      
                val_rev_outputs = val_rev_outputs.view(-1, val_rev_outputs.size(-1))       

                # _, label_predictions = torch.max(outputs, 1)
                val_fwd_targets = val_batch_fwd[1].T
                val_fwd_targets = val_fwd_targets.to(torch.long)   #(seq_len, B = 512)
                val_fwd_targets = val_fwd_targets.reshape(-1)

                val_rev_targets = val_batch_rev[1].T
                val_rev_targets = val_rev_targets.to(torch.long)   #(seq_len, B = 512)
                val_rev_targets = val_rev_targets.reshape(-1)

                # compute fwd and rev losses and backpropogate to train fwd_LSTM and rev_LSTM seperately.
                val_fwd_loss = self.criterion(val_fwd_outputs, val_fwd_targets)
                fwd_validation_loss += val_fwd_loss    

                val_rev_loss = self.criterion(val_rev_outputs, val_rev_targets)
                rev_validation_loss += val_rev_loss 

                # time_e = time.time()
                # print('one batch val time: {}'.format(time_e - time_s))

        average_fwd_validation_loss = fwd_validation_loss / len(self.val_dataloader)
        # log results to text file
        print(f'Validation Loss fwd: {average_fwd_validation_loss:.4f}, Epoch: {epoch_no}')
        wandb.log({'val loss fwd LSTM': average_fwd_validation_loss, 'epoch': epoch_no}) #, 'batch': t})

        average_rev_validation_loss = rev_validation_loss / len(self.val_dataloader)
        # log results to text file
        print(f'Validation Loss rev: {average_rev_validation_loss:.4f}, Epoch: {epoch_no}')
        wandb.log({'val loss rev LSTM': average_rev_validation_loss, 'epoch': epoch_no}) #, 'batch': t})

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



