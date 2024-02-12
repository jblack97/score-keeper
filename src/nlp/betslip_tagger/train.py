import json
import torch
from sklearn.model_selection import train_test_split

from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

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
    for tag_type in ["coarse", "fine"]:
        tag2idx[tag_type] = {
            tag: ind
            for ind, tag in enumerate(data_config.ner_labels.keys())
            if data_config.ner_labels[tag]["type"] in ["default", tag_type]
        }

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
