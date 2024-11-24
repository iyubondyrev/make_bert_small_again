from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer
from datasets import load_dataset

def align_labels_with_tokens(ner_tags: list[int], tokenized_text) -> list[int]:
    aligned_labels = []
    current_word_idx = -1
    prev_word_id = None

    for word_id in tokenized_text.word_ids():
        if word_id is None:
            aligned_labels.append(-100)
        elif word_id != prev_word_id:
            current_word_idx += 1
            aligned_labels.append(ner_tags[current_word_idx])
        else:
            label = ner_tags[current_word_idx]
            aligned_labels.append(label + 1 if label % 2 == 1 else label)

        prev_word_id = word_id

    assert len(tokenized_text.tokens()) == len(aligned_labels)
    
    return aligned_labels


def get_tokenized_dataset():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_and_align_labels(example):
        tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
        aligned_labels = align_labels_with_tokens(example["ner_tags"], tokenized_inputs)
        tokenized_inputs["labels"] = aligned_labels
        return tokenized_inputs

    dataset = load_dataset("eriktks/conll2003")

    dataset = dataset.remove_columns(["id", "pos_tags", "chunk_tags"])
    
    return dataset.map(tokenize_and_align_labels, batched=False)


class FilteredDataCollatorForTokenClassification:
    def __init__(self, tokenizer, label_padding_id=-100):
        self.data_collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=label_padding_id)
    
    def __call__(self, features):
        filtered_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "labels": feature["labels"],
            }
            for feature in features
        ]
        return self.data_collator(filtered_features)