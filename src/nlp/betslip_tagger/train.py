import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from model import BetBERT
from synthetic_dataset.data_config import data_config

MAX_LEN = 75
BATCH_SIZE = 32


def tokenize(sentence, labels):
    tokenized_sentence = []
    coarse_labels = []
    fine_labels = []

    for word, label in zip(sentence, labels):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)
        coarse_labels.extend([label[0]] * n_subwords)
        fine_labels.extend([label[1]] * n_subwords)

    return tokenized_sentence, coarse_labels, fine_labels


if __name__ == "__main__":
    with open(
        "/Users/jblack/projects/score_keeper/src/nlp/betslip_tagger/synthetic_dataset/synthetic_betslips/dataset.json", "r"
    ) as f:
        data = json.loads(f.read())

    tag2idx = {}
    idx2tag = {}
    for tag_type in ["coarse", "fine"]:
        tag2idx[tag_type] = {}
        idx2tag[tag_type] = {}
        for ind, tag in enumerate(data_config.ner_labels.keys()):
            if data_config.ner_labels[tag]["type"] in ["default", tag_type]:
                tag2idx[tag_type][tag] = ind
                idx2tag[tag_type][ind] = tag

    model = BetBERT.from_pretrained(
        "bert-base-cased", num_labels_coarse=5, num_labels_fine=10, output_attentions=False, output_hidden_states=False
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)

    tokenized_data = [tokenize(row["words"], row["labels"]) for row in data]
    tokenized_words = [
        torch.tensor(tokenizer.convert_tokens_to_ids(token_label_pair[0][:MAX_LEN])) for token_label_pair in tokenized_data
    ]
    tokenized_coarse_labels = [
        torch.tensor([tag2idx["coarse"].get(label) for label in token_label_pair[1][:MAX_LEN]])
        for token_label_pair in tokenized_data
    ]
    tokenized_fine_labels = [
        torch.tensor([tag2idx["fine"].get(label) for label in token_label_pair[2][:MAX_LEN]])
        for token_label_pair in tokenized_data
    ]

    tokenized_words = pad_sequence(tokenized_words, batch_first=True, padding_value=0.0)
    tokenized_coarse_labels = pad_sequence(tokenized_words, batch_first=True, padding_value=tag2idx["coarse"]["PAD"])
    tokenized_fine_labels = pad_sequence(tokenized_words, batch_first=True, padding_value=tag2idx["fine"]["PAD"])
    attention_masks = torch.tensor([[float(word != 0.0) for word in sentence] for sentence in tokenized_words])
    X_train, X_val, y_coarse_train, y_coarse_val, y_fine_train, y_fine_val = train_test_split(
        tokenized_words, tokenized_coarse_labels, tokenized_fine_labels, random_state=69, test_size=0.1
    )
    masks_train, masks_val, _, _ = train_test_split(attention_masks, tokenized_words, random_state=69, test_size=0.1)

    train_data = TensorDataset(X_train, masks_train, y_coarse_train, y_fine_train)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
    val_data = TensorDataset(X_val, masks_val, y_coarse_val, y_fine_val)
    val_sampler = RandomSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)

    model = BetBERT.from_pretrained(
        "bert-base-cased", num_labels_coarse=len(tag2idx["coarse"]), num_labels_fine=len(tag2idx["fine"])
    )
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay_rate": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay_rate": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-8)
    epochs = 3
    max_grad_norm = 1.0
    training_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, 0, training_steps)
    device = torch.device("mps")
    train_loss = []
    val_loss = []

    model.to(device)

    for _ in range(epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            X, mask, y_coarse, y_fine = batch
            model.zero_grad()

            outputs = model(X, mask, labels={"coarse": y_coarse, "fine": y_fine})

            outputs.loss.backward()
            total_loss += outputs.loss.item()

            optimizer.step()
            scheduler.step()
            # Calculate the average loss over the training data.

        avg_train_loss = total_loss / len(train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))

        train_loss.append(avg_train_loss)

        # validation
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions, labels, label_ids = {"coarse": [], "fine": []}, {"coarse": [], "fine": []}, {"coarse": [], "fine": []}
        for batch in val_dataloader:
            batch = tuple(t.to(device) for t in batch)
            X, mask, y_coarse, y_fine = batch
            batch_labels = {"coarse": y_coarse, "fine": y_fine}
            with torch.no_grad():
                outputs = model(X, mask, labels={"coarse": y_coarse, "fine": y_fine})
            # Move logits and labels to CPU
            for label_type in ["coarse", "fine"]:
                labels[label_type].extend(batch_labels[label_type].detach().cpu().numpy())
                predictions[label_type].extend(
                    [list(p) for p in np.argmax(outputs.logits[label_type].detach().cpu().numpy(), axis=2)]
                )
            eval_loss += outputs.loss.mean().item()

        eval_loss = eval_loss / len(val_dataloader)
        val_loss.append(eval_loss)
        print("Validation loss: {}".format(eval_loss))
        pred_tags = [
            idx2tag["coarse"][p_i]
            for p, l in zip(predictions, labels["coarse"])
            for p_i, l_i in zip(p, l)
            if idx2tag["coarse"][l_i] != "PAD"
        ]
        valid_tags = [idx2tag["coarse"][l_i] for l in labels["coarse"] for l_i in l if idx2tag[l_i] != "PAD"]
        print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
