# Perform inference with a trained model.
import argparse,json,os,torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from data import CharTokenizer, WsdDataset
from model01 import WsdModelConfig, WsdModel, LemmaTokenizer
import torch.nn.functional as F
from safetensors.torch import load_file
import functions

def load_model_and_tokenizer(args):
    """Load the trained model, tokenizer(s), and vocabularies."""
    
    model_dir = args.model_dir
    data_dir = args.data_dir
    # Load model config and weights
    config_path = os.path.join(model_dir, "config.json")
    model_path = os.path.join(model_dir, "model.safetensors")
    lemma_vocab_path = os.path.join(data_dir, "lemma-vocab.json")
    input_vocab_path = os.path.join(data_dir, "input-vocab.json")
    meaning_emb_path = os.path.join(data_dir, "meaning-embeddings.npy")
    tokenizer_lem_path = os.path.join(data_dir, "lemma-tokenizer-vocab.json")
    
    for p in [config_path,model_path,lemma_vocab_path,input_vocab_path,meaning_emb_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File {p} not found. Exit.")
    
    # Load config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    config = WsdModelConfig(**config_dict)
    
    # Load vocabularies
    with open(lemma_vocab_path, 'r') as f:
        lemma_vocab = json.load(f)
    with open(input_vocab_path, 'r') as f:
        char2id = json.load(f)
    tokenizerSen = CharTokenizer(char2id)

    # Lemma tokenizer (for lem=transformer(word))
    tokLemVoc = {}
    if os.path.exists(tokenizer_lem_path):
        with open(tokenizer_lem_path,'r') as f:
            tokLemVoc = json.load(f)
    tokenizerLem = LemmaTokenizer(lemmata=None,vocab=tokLemVoc)
    
    # Load meaning embeddings
    meaning_embeddings = np.load(meaning_emb_path)
    
    # Create model; not using autoload, because there are additional arguments for the ctr.
    model = WsdModel(config, len(lemma_vocab), config.hidden_size,
                     args.sense_encoding, 
                     meaning_embeddings, False, 
                     args.lemma_encoding,
                     args.max_length_lemma,
                     len(tokenizerLem.char2id),
                     args.sentence_encoding,
                     args.architecture)
    
    # Load model weights
    model.load_state_dict(load_file(model_path))
    
    return model, tokenizerSen, tokenizerLem, lemma_vocab, config

def main():
    parser = argparse.ArgumentParser(description="Apply the WSD model.")
    parser.add_argument("input_file", type=str, help="Input JSONL file")
    parser.add_argument("model_dir", type=str, help="Directory containing the trained model")
    parser.add_argument("data_dir", type=str, help="Directory containing vocabularies")
    parser.add_argument("--lemma_encoding",type=str,
                        default="static",choices=["static","transformer","none"],
                        help="How to treat lemmas whose meanings are predicted?")
    parser.add_argument("--sentence_encoding",type=str,
                        default="full",choices=["full","range"],
                        help="Context represented by the [full] sentence or the string [range] containing the target?")
    parser.add_argument("--sense_encoding",type=str,
                        default="sensim",choices=["sensim","transformer"],
                        help="How to encode the lexicographic definitions?")
    parser.add_argument("--architecture",type=str,
                        default="default",choices=["default","bilinear"],
                        help="Which model architecture?")
    parser.add_argument("--max_length_lemma",type=int,default=64,help="Maximal acceptable length of one lemma in characters")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length")
    args = parser.parse_args()
    
    ix2def = functions.loadIx2Def(args)

    # Determine output file path
    input_path = args.input_file
    output_path = input_path[:-6] + '-classified.jsonl'
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    try:
        model, tokenizerSen, tokenizerLem, lemma_vocab, _ = load_model_and_tokenizer(args)
        if args.lemma_encoding=='transformer' and len(tokenizerLem.char2id)==0:
            print(f"Lemma encoding mode {args.lemma_encoding} requires a lemma tokenizer, but its vocabulary is empty.")
            return
        if args.lemma_encoding=="static":
            if len(lemma_vocab)==0:
                print("Couldn't load the lemma vocabulary.")
                return
            print(f"{len(lemma_vocab)} items in the lemma vocabulary")
        print("Model loaded successfully")
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        return
    
    # Load dataset
    print("Loading dataset...")
    try:
        raw_dataset = load_dataset("json", data_files={"test": input_path})["test"]
        print(f"Loaded {len(raw_dataset)} records")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    max_num_defs = max(len(item["embIxes"]) for item in raw_dataset)
    print(max_num_defs)
    dataset = WsdDataset(
        raw_dataset,
        tokenizerSen, tokenizerLem, model.definition_tokenizer,
        lemma_vocab, 
        args.max_seq_length, max_num_defs, 
        temperature=0,
        sense_encoding_method=args.sense_encoding,
        ix2def=ix2def
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Apply model
    print("Applying model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    ix = 0 # Index in input
    with open(output_path, 'w', encoding='utf-8') as f:
        with torch.no_grad():
            for batch in dataloader:
                # Forward pass = Model.forward(...)
                # TODO add split lemmas!
                outputs = model(
                    lemma_id=batch["lemma_id"].to(device), # one id per batch item
                    input_ids=batch["input_ids"].to(device), # chars in the sentence
                    attention_mask=batch["attention_mask"].to(device), # for the sentence
                    meaning_ids=batch["meaning_ids"].to(device), # senses per batch item
                    input_ids_lemma = batch["input_ids_lemma"].to(device),
                    attention_mask_lemma = batch["attention_mask_lemma"].to(device),
                    input_ids_def = batch["input_ids_def"].to(device),
                    attention_mask_def = batch["attention_mask_def"].to(device),
                    ranges = batch["ranges"].to(device)
                )
                
                logits = outputs["logits"]  # (batch_size, max_senses)
                
                # Process each item in the batch
                batch_size = logits.size(0)
                for i in range(batch_size):
                    # Get number of valid senses from original data, and truncate to valid senses.
                    item_logits = logits[i, :len(raw_dataset[ix]["embIxes"])]
                    probs = F.softmax(item_logits, dim=0).cpu().tolist()
                    d = dict(raw_dataset[ix]) # Creates a copy, everything else identical.
                    d["predictions"] = probs
                    # Write to file.
                    f.write(json.dumps(d, ensure_ascii=False) + '\n')
                    # Increase the global counter.
                    ix+=1
    print(f"Results written to {output_path}")
    
if __name__ == "__main__":
    main()