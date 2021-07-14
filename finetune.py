import pandas as pd
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, AutoConfig, get_linear_schedule_with_warmup
import numpy as np
import random
import os
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import jsonlines
import json
from pandas import DataFrame


# 0. config
# Subject for data path: Biology, Law, Psychology, Merged_dataset
data_dir = "./MNLI"

# 'Num_options', 'Tags', 'NW_S', 'NW_A', 'NW_D0', 'NW_D1', 'NW_D2', 'NW_D3', 'NW_E',
# 'Ambiguity', 'Grammar', 'Readability', 'Plausible_distractor'
#handcrafted_features = ['Grammar', 'NW_E']

pretrained_model_name = "RoBERTa"  # pretrained model: BERT, BioBERT, RoBERTa, paraphrase-distilroberta-base-v1

# validation dataset proportion from train.xlsx
#val_size = 0.1111  # (e.g: when train:test = 9:1, to make train:val:test = 8:1:1, the val_size should be 1/9)
batch_size = 16  # for DataLoader (when fine-tuning BERT on a specific task, 16 or 32 is recommended)
epochs = 10  # Number of training epochs (we recommend between 2 and 4)
lr = 1e-5  # Optimizer parameters: learning_rate - default is 5e-5, our notebook had 2e-5
eps = 1e-8  # Optimizer parameters: adam_epsilon  - default is 1e-8.
num = 3 # the parameter to the num_labels

seed_val = 2021  # Set the seed value all over the place to make this reproducible.
saved_model_dir = "./checkpoints/roberta_large_mnli_local/"  # save model after fine-tune


# Set logger to avoid warning `token indices sequence length is longer than the specified maximum sequence length for this model (1017 > 512)`
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def text_to_id(tokenizer, text_list):
    """
    It is a function to transform text to id.
    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #
    # Explanation from another place:
    # BERT needs adding special tokens --- [cls] and [sep]
    #
    # The tokenizer.encode function combines multiple steps for us:
    #
    # Split the sentence into tokens.
    # Add the special [CLS] and [SEP] tokens.
    # Map the tokens to their IDs.
    # Oddly, this function can perform truncating for us, but doesn't handle padding.
    #
    # Encoding for text in training dataset
    """
    ids_list = []

    for item in text_list:
        # Sentence to id and add [CLS] and [SEP]
        encoded_item = tokenizer.encode(item, add_special_tokens=True)
        ids_list.append(encoded_item)

    return ids_list


def padding_truncating(input_ids_list, max_length):
    """
    It is a function to perform padding and truncating
    @param input_ids_list: <List> text_ids
    @param max_length: <Integer> the number we wanna the sentence to be padding or truncating
    @return: processed input_ids_list
    """
    processed_input_ids_list = []
    for item in input_ids_list:
        seq_list = []

        if len(item) < max_length:
            # Define a seq_list with the length of max_length
            seq_list = [0] * (max_length - len(item))
            item = item + seq_list

        elif len(item) >= max_length:
            item = item[:max_length]

        processed_input_ids_list.append(item)

    return processed_input_ids_list


def get_attention_masks(pad_input_ids_list):
    """
    It is a function to get attention masks:

    - If a token ID is 0, then it's padding, set the mask to 0.
    - If a token ID is > 0, then it's a real token, set the mask to 1.
    """
    attention_masks_list = []

    for item in pad_input_ids_list:

        mask_list = []
        for subitem in item:
            if subitem > 0:
                mask_list.append(1)
            else:
                mask_list.append(0)
        attention_masks_list.append(mask_list)

    return attention_masks_list


def load_dataset(data_path):
    cols = ['Rating', 'Stem', 'Answer', 'Distractor0', 'Distractor1', 'Distractor2', 'Distractor3', 'Explanation']
    data_frame = pd.read_excel(data_path, sheet_name="Sheet1", usecols=cols)
    dataset = data_frame.values
    # split into input (x) and output (y) variables; skip index at column 0
    x = dataset[:, 1:]
    y = dataset[:, 0]

    x = x.astype(str)
    x = [" ".join(i) for i in x]

    return x, list(y)

def load_dataset_mnli(data_path):
    df = DataFrame(columns=['gold_label', 'sentence1', 'sentence2'])
    with open(data_path, 'r', encoding="utf-8") as f:
        for jsonstr in f.readlines():
            jsonstr = json.loads(jsonstr)
            label = 0
            if jsonstr['gold_label'] == 'entailment':
                label = 0
            elif jsonstr['gold_label'] == 'neutral':
                label = 1
            elif jsonstr['gold_label'] == 'contradiction':
                label = 2
            result_row = {'gold_label': label,
                          'sentence1': jsonstr['sentence1'],
                          'sentence2': jsonstr['sentence2']}
            df = df.append(result_row, ignore_index=True)
    cols = ['gold_label', 'sentence1', 'sentence2']
    dataset = df.values
    # split into input (x) and output (y) variables; skip index at column 0
    x = dataset[:, 1:]
    y = dataset[:, 0]

    x = x.astype(str)
    x = [" ".join(i) for i in x]

    return x, list(y)

# TODO add main method, modulize all source code
if __name__=="__main__":
    # 1. Data Pre-processing
    # load data
    train_text, train_labels = load_dataset_mnli(data_dir + "/multinli_1.0_train.jsonl")
    val_text, val_labels = load_dataset_mnli(data_dir + "/multinli_1.0_dev.jsonl")
    #test_text, test_labels = load_dataset_pair(data_dir + "/threshold-0.5 Biology rating similarity test pair.xlsx")

    # 2. BERT Tokenization & Input Formatting
    # 2.1 BERT Tokenization
    # Load the BERT tokenizer.
    if pretrained_model_name == "BERT":
        pretrained_model = "bert-base-uncased"
    elif pretrained_model_name == "BioBERT":
        pretrained_model = "dmis-lab/biobert-v1.1"
    elif pretrained_model_name == "RoBERTa":
        pretrained_model = "roberta-large"
    elif pretrained_model_name == "paraphrase-distilroberta-base-v1":
        pretrained_model = "sentence-transformers/paraphrase-distilroberta-base-v1"
    else:
        pretrained_model = "bert-base-uncased"

    print('Loading ' + pretrained_model_name +' tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, do_lower_case=True)

    ## 2.2 Input Formatting for BERT
    train_text_ids = text_to_id(tokenizer, train_text)
    val_text_ids = text_to_id(tokenizer, val_text)
    #test_text_ids = text_to_id(tokenizer, test_text)

    ## 2.3 Padding & Truncating
    ## Padding or truncating the train_text_ids and test_text_ids
    # TODO QQ suggest using the default padding method
    train_padding_list = padding_truncating(train_text_ids, max_length=512)
    val_padding_list = padding_truncating(val_text_ids, max_length=512)
    #test_padding_list = padding_truncating(test_text_ids, max_length=512)

    ## 2.4 Attention Masks
    train_attention_masks = get_attention_masks(train_padding_list)
    val_attention_masks = get_attention_masks(val_padding_list)
    #test_attention_masks = get_attention_masks(test_padding_list)

    ## 2.6 Convert to Dataset
    ## 2.6.1 Convert all the List objects to tensor
    # Convert all inputs and labels into torch tensors, the required datatype for our model.
    train_inputs = torch.tensor(train_padding_list)
    validation_inputs = torch.tensor(val_padding_list)
    #test_inputs = torch.tensor(test_padding_list)

    train_labels = torch.LongTensor(train_labels)
    validation_labels = torch.LongTensor(val_labels)
    #test_labels = torch.FloatTensor(test_labels)

    train_masks = torch.tensor(train_attention_masks)
    validation_masks = torch.tensor(val_attention_masks)
    #test_masks = torch.tensor(test_attention_masks)

    ## 2.6.2 Form the Dataset with torch.tensor
    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = RandomSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # Create the DataLoader for our test set.
    # test_data = TensorDataset(test_inputs, test_masks, test_labels)
    # test_sampler = SequentialSampler(test_data)
    # test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)


    # 3. Train BERT Text Classification Model
    # 3.1 AutoModelForSequenceClassification
    # Load AutoModelForSequenceClassification from transformers
    # You can increase this for multi-class tasks.
    # https://discuss.huggingface.co/t/which-loss-function-in-AutoModelForSequenceClassification-regression/1432/2
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model,
        num_labels=num,                   # The number of output labels -- 1 for MSE Loss Regression.
        output_attentions=False,        # Whether the model returns attentions weights.
        output_hidden_states=False,     # Whether the model returns all hidden-states.
    )
    model.to(device)

    ## 3.2 Optimizer & Learning Rate Scheduler
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(), lr=lr, betas = (0.9, 0.98), eps=eps)

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    print("total_steps = {}".format(total_steps))

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    ## 3.3 Train
    # Function to calculate the MSE of our predictions and ground truth
    def flat_mse(preds, labels):
        pred_flat = preds.flatten()
        labels_flat = labels.flatten()
        return np.sum((pred_flat - labels_flat)**2) / len(labels_flat)


    def format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))


    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []
    eval_loss_values = []

    # For each epoch...
    for epoch_i in range(epochs):

        ##########################################
        #               Training                 #
        ##########################################

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 10 batches.
            if step % 10 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Clear the gradients.
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we have provided the `labels`.
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple.
            loss = outputs[0]

            # Accumulate the training loss over all of the batches so that we can calculate the average loss at the end.
            # `loss` is a Tensor containing a single value; the `.item()` function just returns the Python value from the tensor.
            total_loss += loss.item()

            # Perform a `backward` pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        ##########################################
        #               Validation               #
        ##########################################
        # After the completion of each training epoch, measure our performance on our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
        model.eval()

        # Tracking variables
        eval_loss, eval_mse = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to device
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have not provided labels.
                # token_type_ids is the same as the "segment ids", which differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.AutoModelForSequenceClassification
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)

            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu()
            label_ids = b_labels.to('cpu')

            if num > 1:
                loss_func = torch.nn.CrossEntropyLoss()
                loss = loss_func(logits, label_ids)
            else:
                logits = logits.numpy()
                label_ids = label_ids.numpy()
                # Calculate the mse for this batch of test sentences.
                loss = mean_squared_error(logits, label_ids)

            # Accumulate the total mse.
            eval_loss += loss

            # Track the number of batches
            nb_eval_steps += 1

        average_val_loss = eval_loss / len(validation_dataloader)

        eval_loss_values.append(average_val_loss)
        # Report the final accuracy for this validation run.
        print("  MSE: {0:.2f}".format(average_val_loss))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

        saved_epoch_model = saved_model_dir + "epoch_" + str(epoch_i) + "/"

        if not os.path.exists(saved_epoch_model):
            os.makedirs(saved_epoch_model)

        # Save model to the saved_epoch_model
        model.save_pretrained(saved_epoch_model)
        tokenizer.save_pretrained(saved_epoch_model)

    print("Training complete!")

    ## 3.4 Plot
    ## Plot the average loss in training

    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)
    x_values=list(range(epochs))
    # Plot the learning curve.
    plt.plot(x_values,loss_values, 'b-o', label='Training MSE')
    plt.plot(x_values,eval_loss_values, 'rs-', label='Validation MSE')
    plt.legend()
    # Label the plot.
    plt.title("Training and validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(saved_model_dir + pretrained_model_name + ".jpg")
    plt.show()


    # 3.6 Saving Trained Model
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)

    # Save model to the saved_model_dir
    model.save_pretrained(saved_model_dir)
    tokenizer.save_pretrained(saved_model_dir)
