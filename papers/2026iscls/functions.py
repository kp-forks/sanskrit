import json,os

def buildLemmaVocabulary(paths):
    print("Building lemma vocabulary...")
    # The current model does not need a padding voc id, because this vocabulary
    # is only used for single words, not for sequences.
    vocab = {} # {"<pad>" : constants.PAD_LABEL_ID}
    maxNDefs = 0
    for p in paths:
        with open(p, 'rb') as f:
            for u,lineRaw in enumerate(f):
                try:
                    line = lineRaw.decode("utf-8")
                    if line.strip():
                        obj = json.loads(line)
                        lemma = obj["lemma"]
                        if not lemma in vocab:
                            vocab[lemma] = len(vocab)
                        m = obj["embIxes"]
                        if len(m) > maxNDefs:
                            maxNDefs = len(m)
                except UnicodeDecodeError:
                    print(f"Skipping line {u} due to encoding error.")
                except json.JSONDecodeError as jde:
                    print(f"[JSON Error] {p} line {u}: {jde} -> {lineRaw.decode('utf-8', errors='replace')}")
                except Exception as e:
                    print(f"[Other Error] {p} line {u}: {e}")
    return vocab,maxNDefs

def loadIx2Def(args):
    ix2def = {}
    if args.sense_encoding=="transformer":
        path = f"{args.data_dir}/def2ix.json"
        if not os.path.exists(path):
            print("def2ix.json not found in the data directory")
            exit()
        with open(path,"r",encoding="UTF-8") as f:
            m = json.load(f)
            for s,i in m.items():
                ix2def[i] = s # mapping index --> definition
    return ix2def