import argparse,json,os
import functions,constants
from data import CharTokenizer,WsdDataset
from model01 import WsdModelConfig,WsdModel,LemmaTokenizer
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, TrainerCallback, PretrainedConfig
from sklearn.metrics import precision_score, recall_score, f1_score
from safetensors.torch import load_file
import numpy as np


def prepare_compute_metrics(nSenses):
    def compute_eval_metrics(pred):
        # Filter to obtain only multi-sense records
        msm = np.array([n > 1 for n in nSenses])
        # Edge case: all records are single-sense
        if not msm.any():
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
        
        # Get predictions and labels, filtered to multi-sense only
        preds = pred.predictions.argmax(axis=-1)[msm]
        labels = pred.label_ids[msm]
        
        # Filter out any padded examples (if needed)
        mask = labels != constants.PAD_LABEL_ID
        preds_filtered = preds[mask]
        labels_filtered = labels[mask]
        
        return {
            "precision": precision_score(labels_filtered, preds_filtered, average='weighted', zero_division=0),
            "recall": recall_score(labels_filtered, preds_filtered, average='weighted', zero_division=0),
            "f1": f1_score(labels_filtered, preds_filtered, average='weighted', zero_division=0),
            "accuracy": (preds_filtered == labels_filtered).mean().item()
        }
    return compute_eval_metrics

def compute_metrics(pred):
    preds = pred.predictions.argmax(axis=-1)  # (batch,) - predicted class indices; pred.predictions are the logits.
    labels = pred.label_ids          # (batch,) - true class indices
    
    # Filter out any padded examples; probably not needed for WSD
    mask = labels != constants.PAD_LABEL_ID
    labels_filtered = labels[mask]
    preds_filtered = preds[mask]
    
    return {
        "precision": precision_score(labels_filtered, preds_filtered, average='weighted', zero_division=0),
        "recall": recall_score(labels_filtered, preds_filtered, average='weighted', zero_division=0),
        "f1": f1_score(labels_filtered, preds_filtered, average='weighted', zero_division=0),
        "accuracy": (preds_filtered == labels_filtered).mean().item()
    }

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file", type=str)
    parser.add_argument("eval_file", type=str)
    parser.add_argument("meaning_emb_file",type=str,help=".npy file with the precalculated meaning embeddings.")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=256,help="Same as for the pretrained model.")
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--num_hidden_layers",type=int, default=4)
    parser.add_argument("--num_attention_heads",type=int,default=4)
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
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--eval_steps", type=int, default=20000)
    parser.add_argument("--output_dir", type=str, default="./model")
    parser.add_argument("--pretrained_model_dir",type=str,default="",help="If train_mode=finetune, this is the directory of the pretrained model.")
    parser.add_argument("--pretrained_model_data_dir",type=str,default="",help="If train_mode=finetune, this is the data directory of the pretrained model (vocabulary file).")
    parser.add_argument("--temperature",type=float,default=0.0,help="If >0, soft targets are calculated using the temperature.")
    args = parser.parse_args()

    # Sanity
    for s in [args.train_file,args.eval_file,args.meaning_emb_file]:
        if not os.path.exists(s):
            print(f"The critical file {s} does not exist.")
            exit()
    if not os.path.exists(os.path.join(args.pretrained_model_dir, "config.json")):
        print("Pretrained config file missing.")
        exit()
    
    # Meaning embeddings.
    mEmbs = np.load(args.meaning_emb_file)

    # Indices --> lexicographic definitions as strings.
    ix2def = functions.loadIx2Def(args)

    # Lemma vocabulary.
    vocab,maxNDefs = functions.buildLemmaVocabulary([args.train_file,args.eval_file])
    print(f"Vocabulary size: {len(vocab)}, maximal number of definitions: {maxNDefs}.")
    # Tokenizer for lemmas
    tokenizerLem = LemmaTokenizer(vocab,None,args.max_length_lemma)
    print(f"Lemma tokenizer vocabulary size: {len(tokenizerLem.char2id)}")
    # Store the vocabulary for later application.
    with open(f"{args.data_dir}/lemma-vocab.json", "w", encoding='utf-8') as f:
        json.dump(vocab, f, indent=4)
    with open(f"{args.data_dir}/lemma-tokenizer-vocab.json","w",encoding="utf-8") as f:
        json.dump(tokenizerLem.char2id,f,indent=4)

    # Plain text vocabulary: reused from the pretrained model.
    # This is for the sentence transformer.
    vocabPath = f"{args.pretrained_model_data_dir}/input-vocab.json"
    with open(vocabPath, 'r') as f:
        char2id = json.load(f)
        tokenizerSen = CharTokenizer(char2id)
        # Save a copy in the real data directory.
        with open(f"{args.data_dir}/input-vocab.json", "w") as f:
            json.dump(tokenizerSen.char2id, f, indent=4)
    
    print("Loading dataset...")
    raw_datasets = load_dataset("json", data_files={"train": args.train_file, "validation": args.eval_file})
    
    # 1-sense cases in eval
    nEvalSenses = [len(d["embIxes"]) for d in raw_datasets["validation"]]

    
    # AutoConfig does not work here, because this model is not registered.
    with open(os.path.join(args.pretrained_model_dir, "config.json")) as f:
        configRaw = json.load(f)
    config = WsdModelConfig(**configRaw)
    model = WsdModel(config,len(vocab),
                     args.hidden_size,
                     args.sense_encoding,
                     mEmbs,
                     args.temperature > 0,
                     args.lemma_encoding,
                     args.max_length_lemma,
                     len(tokenizerLem.char2id),
                     args.sentence_encoding,
                     args.architecture)
    # Load the pretrained Sanskrit sentence transformer.
    state_dict = load_file(os.path.join(args.pretrained_model_dir, "model.safetensors"))
    # Filter out classifier head weights; use "classifier", because this is the name of the classification head in the model.
    filtered_state_dict = {
        k: v for k, v in state_dict.items() if not k.startswith("classifier")
    }
    
    # Load filtered weights into classifier model
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    print(f"   Loaded pretrained weights (excluding head).")
    print(f"   Missing keys: {missing_keys}")
    print(f"   Unexpected keys: {unexpected_keys}")

    # Data must be loaded here because they may require the SentenceTransformer tokenizer loaded by the main model.
    train_ds = WsdDataset(raw_datasets["train"], 
                          tokenizerSen,tokenizerLem, 
                          model.definition_tokenizer, vocab, 
                          args.max_seq_length, 
                          maxNDefs,args.temperature,
                          args.sense_encoding,ix2def)
    eval_ds = WsdDataset(raw_datasets["validation"], tokenizerSen,tokenizerLem, model.definition_tokenizer, vocab, 
                         args.max_seq_length, maxNDefs,args.temperature,args.sense_encoding,ix2def)
    print(f"Train: {len(train_ds.dataset)}, eval: {len(eval_ds.dataset)}, eval senses: {len(nEvalSenses)}")

    print("Starting training...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size= args.batch_size, # ... we have a large vocabulary!!
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="linear",
        fp16=True, # <<<<< !! only for sense_encoding="transformer"
        eval_strategy="steps", # <<<<<<!!!! "eval_strategy" for later versions of transformers.
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        logging_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",  # Disable wandb/huggingface logging unless desired
        eval_accumulation_steps=1,  # Move eval data to CPU to avoid OOM.
        label_names=["labels"] ## !!!!! <<< THIS fixed the issue with eval_f1 not found etc. Not sure why it was not needed in the initial version of the training script.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=prepare_compute_metrics(nEvalSenses)
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print("Training complete. Model saved to:", args.output_dir)


if __name__ == "__main__":
    main()
