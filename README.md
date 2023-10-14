# Problem 1: NNLM
In this scenario, Neural network language model is trained by varying dropout rates from (0.3 to 0.8) and also hidden layer size. Perplexity scores for all the models are reported below. 

Training script: source/Problem_1_NNLM.py

# Problem 2: LSTM
In this scenario, Neural network language model is trained by varying hidden layer size (300 and 500). Perplexity scores are reported below along with checkpoints.  

Training script: source/Problem_2_LSTM.py

Checkpoints (pretrained models) for all the below results are mentioned below (shared in gdrive).

Checkpoints for all models (NNLM and also LSTM) : https://drive.google.com/drive/folders/1577zJiFL5Pks23DS36HkKuePcDIatsAZ?usp=sharing

## Loading checkpoints and get perplexity scores: 
Run script (report_perplexity_scores.py) in source folder and provide the checkpoint path in Line (193) and file name in (197 and 203)

## Results 

 Model |   Checkpoints        
-------|-----------------------------------------------------|
NNLM (Best model =  dropout : 0.3, hiiden1:300, hidden2:300) |  https://drive.google.com/file/d/1tadHF1y8b42pbIEK-LZ7zPCWw11r6OrQ/view?usp=drive_link     | 
LSTM  (hidden layer = 300)|  https://drive.google.com/file/d/18k9BWya7xhscytCvZPLN4440z9TwvhfX/view?usp=sharing     | 

## Perplexity files (Folder: Perplexity_scores): 
All the perplexity files for both the models (train, val, test) are in folder: Perplexity_scores

## Loss plots: 
* NNLM: NNLM_training_loss_plots.png
* LSTM: LSTM_training_loss_plots.png

## Contact
* General
>Jayaram Reddy<jayaram.reddy@research.iiit.ac.in><br />
