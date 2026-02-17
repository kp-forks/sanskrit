from torch.utils.data import Dataset
import torch
import constants
import numpy as np

class CharTokenizer:
    def __init__(self, vocab: dict):
        # Reuse an existing vocabulary, either for application, or when finetuning a pretrained model.
        self.char2id = vocab
        self.id2char = {v: k for k, v in self.char2id.items()}
        self.pad_token_id = 0
        self.unk_token_id = 1
        # The next symbol may not be contained in all versions.
        self.mask_token_id = self.char2id["[MASK]"] if "[MASK]" in self.char2id else -1

    def __call__(self, text, max_length=256):
        input_ids = [self.char2id.get(c, self.unk_token_id) for c in text[:max_length]]
        attention_mask = [1] * len(input_ids)

        # Padding
        pad_len = max_length - len(input_ids)
        input_ids += [0] * pad_len
        attention_mask += [0] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def batch_encode(self, texts, max_length=256):
        return [self(text, max_length) for text in texts]

    def vocab_size(self):
        return len(self.char2id)


class WsdDataset(Dataset):
    '''
    Data set for training or finetuning.
    '''
    def __init__(self, hf_dataset, tokenizerSen, tokenizerLemma, tokenizerDef, vocabLemma: dict, max_length_sen: int, max_num_defs: int, 
                 temperature: float,
                 sense_encoding_method : str,
                 ix2def : dict):
        self.dataset = hf_dataset
        self.tokenizerSentence = tokenizerSen
        self.tokenizerLemma = tokenizerLemma
        self.tokenizerDef = tokenizerDef # only not None if sense_encoding_method="transformer"
        self.vocabLemma = vocabLemma
        self.max_length = max_length_sen
        self.maxNumDefs = max_num_defs
        self.temperature = temperature
        self.sense_encoding_method = sense_encoding_method
        self.ix2def = ix2def

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        '''
        One record per call!
        '''
        item = self.dataset[idx]
        #Note: IAST --> HK done in R.
        tokens_sen = self.tokenizerSentence(item["sen"], max_length=self.max_length)
        tokens_lem = self.tokenizerLemma(item["lemma"])
        # Convert lemma to id(s)
        # lemmaId = self.vocabLemma[item["lemma"]]
        # Note: the eval set may contain lemmas that are not in train/test.
        lemmaId = self.vocabLemma[item["lemma"]] if item["lemma"] in self.vocabLemma else 0
        
        # Possible meanings
        tokens_def = None
        # We must pass a tensor for the next two values. Leaving them at None produces an error.
        def_ids = torch.tensor([0],dtype=torch.long)
        def_attn= torch.tensor([0],dtype=torch.long)
        #if self.sense_encoding_method=="sensim":
        # Always! We need this to get the right dimension and masking in the model
        meaningIxes = item["embIxes"] + [constants.PAD_LABEL_ID] * (self.maxNumDefs - len(item["embIxes"]))
        if self.sense_encoding_method=="transformer":
            # Fully trainable
            if self.tokenizerDef is None:
                print("No sense tokenizer.")
                exit()
            # Todo: write the code: item["embIxes"] --> lookup in self.ix2def --> tokenize the obtained strings with the model's tokenizer
            defs = []
            for ix in item["embIxes"]:
                if not ix in self.ix2def:
                    print(f"Sense with index {ix} not in the internal mapping. Exit.")
                    exit()
                defs.append(self.ix2def[ix]) # This is a string.
            defs.extend([""] * (self.maxNumDefs - len(defs))) # Padding to the correct length, empty string as dummy.
            tokens_def = self.tokenizerDef(
                defs,
                padding="max_length", #padding=True,
                truncation=True,
                max_length=128, # !! This may be problematic, since the longest definition is 254 chars ... but what about multi-byte chars?
                return_tensors='pt'
            )
            def_ids = tokens_def["input_ids"]
            def_attn= tokens_def["attention_mask"]
        
        
        if self.temperature > 0: # soft targets
            # This branch is not actively developed. Ignore it.
            e = np.exp(-(1.0 - np.array(item["sims"]))/self.temperature)
            return {
                "input_ids": torch.tensor(tokens_sen["input_ids"], dtype=torch.long), # --> transformer
                "attention_mask": torch.tensor(tokens_sen["attention_mask"], dtype=torch.long), # for the sentence
                "input_ids_lemma" : torch.tensor(tokens_lem["input_ids"], dtype=torch.long), # --> lemma transformer
                "attention_mask_lemma" : torch.tensor(tokens_lem["attention_mask"],dtype=torch.long),
                "meaning_ids" : torch.tensor(meaningIxes,dtype=torch.long),
                "lemma_id" : torch.tensor(lemmaId,dtype=torch.long),
                "soft_labels" : torch.tensor(e / np.sum(e), dtype=torch.float32),
                "label": torch.tensor(item["target"].index(1), dtype=torch.long)
                }
        else:
            # one-hot targets
            return {
                "input_ids": torch.tensor(tokens_sen["input_ids"], dtype=torch.long), # go into the transformer
                "attention_mask": torch.tensor(tokens_sen["attention_mask"], dtype=torch.long), # for the sentence
                "input_ids_lemma" : torch.tensor(tokens_lem["input_ids"], dtype=torch.long),
                "attention_mask_lemma" : torch.tensor(tokens_lem["attention_mask"],dtype=torch.long),
                "input_ids_def" : def_ids,
                "attention_mask_def" : def_attn,
                "meaning_ids" : torch.tensor(meaningIxes,dtype=torch.long),
                "lemma_id" : torch.tensor(lemmaId,dtype=torch.long),
                "labels": torch.tensor(item["target"].index(1), dtype=torch.long) if "target" in item else torch.ones(self.maxNumDefs, dtype=torch.long),
                "ranges" : torch.tensor([item["rangeStart"],item["rangeEnd"]],dtype=torch.long)
                }
    

