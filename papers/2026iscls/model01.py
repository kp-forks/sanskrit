import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from sentence_transformers import SentenceTransformer

class LemmaTokenizer:
    """
    This class char-tokenizes the lemmata.
    """
    def __init__(self, lemmata : dict, vocab : dict = None, max_length : int =64):
        self.max_length = max_length
        self.char2id = {}
        if vocab is None:
            # Initial construction
            self.char2id["[PAD]"] = 0
            self.char2id["[UNK]"] = 1
            self.char2id["[MASK]"] = 2
            for w in lemmata:
                for c in w:
                    if not c in self.char2id:
                        self.char2id[c] = len(self.char2id)
        else:
            # Reuse an existing vocabulary, either for application, or when finetuning a pretrained model.
            self.char2id = vocab
        self.id2char = {v: k for k, v in self.char2id.items()}
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.mask_token_id = self.char2id["[MASK]"] if "[MASK]" in self.char2id else -1

    def __call__(self, text):
        input_ids = [self.char2id.get(c, self.unk_token_id) for c in text[:self.max_length]]
        attention_mask = [1] * len(input_ids)

        # Padding
        pad_len = self.max_length - len(input_ids)
        input_ids += [self.char2id["[PAD]"]] * pad_len
        attention_mask += [0] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def batch_encode(self, texts):
        return [self(text) for text in texts]

    def vocab_size(self):
        return len(self.char2id)


"""
Alternative to static lemma embeddings: mini-transformer.
"""
class LemmaEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_layers=2, num_heads=4, ff_dim=512, max_len=64):
        super().__init__()
        self.char_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids, attention_mask):
        # input_ids: [batch, seq_len]
        # attention_mask: [batch, seq_len] with 1 = keep, 0 = pad

        device = next(self.parameters()).device
        input_ids = input_ids.to(device=device) #, dtype=torch.long)
        attention_mask = attention_mask.to(device=device)

        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.char_emb(input_ids) + self.pos_emb(positions)

        src_key_padding_mask = (attention_mask == 0)
        
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.layer_norm(x)

        # Mean pooling (masking out padding tokens)
        lengths = attention_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0) # avoid division by zero
        x = (x * attention_mask.unsqueeze(-1).float()).sum(dim=1) / lengths

        return x  # [batch, embed_dim]

class WsdModelConfig(PretrainedConfig):
    def __init__(self, 
                 vocab_size=100, # Problems when no default is given.
                 hidden_size=256, num_hidden_layers=4,
                 num_attention_heads=4, max_position_embeddings=256, 
                 num_labels = 2,
                 **kwargs):
        """
        vocab_size:  of the sentence transformer, i.e. characters
        """
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.num_labels = num_labels

class WsdModel(PreTrainedModel):

    config_class = WsdModelConfig

    def __init__(self, config, vocab_lemma_size: int, embedding_dim: int, sense_encoding_method : str, 
                 sense_embeddings, 
                 useSoftTargets : bool, 
                 lemma_encoding : str, max_len_lemma : int, lemma_char_vocab_size: int,
                 sentence_encoding : str,
                 architecture : str):
        """

        """
        super().__init__(config)
        
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size) # These weights will be overwritten by those of the pretrained model
        self.useSoftTargets = useSoftTargets
        self.lemma_encoding = lemma_encoding
        self.sense_encoding = sense_encoding_method
        self.sense_embedding_dim = sense_embeddings.shape[1]
        self.sentence_encoding = sentence_encoding
        self.architecture = architecture

        print(f"Lemma encoding: {self.lemma_encoding}")

        # Pass the whole sentence through a transformer.
        self.position_embeddings_sen = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Data structures for encoding lemmas
        self.lemma_embeddings = None
        self.lemma_encoder = None
        if self.lemma_encoding=="static":
            self.lemma_embeddings = nn.Embedding(vocab_lemma_size, embedding_dim)
        elif self.lemma_encoding=="transformer":
            self.lemma_encoder = LemmaEncoder(lemma_char_vocab_size,embed_dim=embedding_dim,max_len=max_len_lemma)

        # Encoding definitions
        # Trainable sense embeddings
        self.definition_encoder = None
        self.definition_tokenizer = None
        if self.sense_encoding=="sensim":
            # From the semantic similarity model
            self.sense_embedding_matrix = nn.Parameter(torch.tensor(sense_embeddings, dtype=torch.float32))
        elif self.sense_encoding=="transformer":
            self.definition_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.definition_tokenizer = self.definition_encoder[0].tokenizer # This is passed to the WsdDataset
            self.sense_projection2 = nn.Sequential(
                nn.Linear(
                    self.definition_encoder.get_sentence_embedding_dimension(),
                    self.sense_embedding_dim # should map from the tokenizer embedding size to self.sense_embedding_dim
                ),
                nn.ReLU()
            )
        else:
            print(f"Unsupported sense embedding method: {self.sense_encoding}")
            exit()
        

        # The next elements are for the Sanskrit sentence.
        self.conv1d_3 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1)
        self.conv1d_5 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=5, padding=2)
        self.projection = nn.Linear(3 * config.hidden_size, config.hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)

        d = self.sense_embedding_dim # sense embedding dimensionality
        # This projection is responsible for projecting the concatenated word and sentence tensors to the right embedding dimension.
        if self.lemma_encoding=="none" or self.architecture=="bilinear":
            # Todo: rename this to sentence_projection.
            self.meaning_projection = nn.Sequential(
                nn.Linear(
                    config.hidden_size, # sentence transformer
                    d),
                nn.ReLU(), # tanh etc are bad ideas.
                nn.LayerNorm(d)
            )
        else:
            self.meaning_projection = nn.Sequential(
                nn.Linear(
                    embedding_dim + # target lemma
                    config.hidden_size, # sentence transformer
                    d),
                nn.ReLU(), # tanh etc are bad ideas.
                nn.LayerNorm(d)
            )
        # Affine: static meanings
        self.sense_projection = nn.Sequential(
            nn.Linear(d,d), # keep the same dimension
            nn.ReLU(),
            nn.LayerNorm(d)
        )
        if self.architecture=="bilinear":
            self.bilinear_projection = nn.Sequential(
                nn.Linear(embedding_dim, d*d),
                nn.Tanh(),
                nn.LayerNorm(d*d)
            )
        

    def encode_sentence(self, input_ids, ranges, attention_mask=None):
        '''
        Encodes the Sanskrit sentence: input -> cnn -> transformer -> penult
        '''
        positions = torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.embeddings(input_ids) + self.position_embeddings_sen(positions)

        # CNN branch
        cnn_3 = self.conv1d_3(x.transpose(1, 2)).transpose(1, 2)
        cnn_5 = self.conv1d_5(x.transpose(1, 2)).transpose(1, 2)
        x = torch.cat([x, cnn_3, cnn_5], dim=-1)
        x = self.projection(x)

        x = x.transpose(0, 1)
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        ctx_vecs = x.transpose(0, 1) # (batch_size, seq_len, hidden_size)

        #return x  # (batch_size, seq_len, hidden_size)
        # Average over non-padded tokens
        if self.sentence_encoding=="full":
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(ctx_vecs).float()
            ctx_masked = ctx_vecs * mask_expanded
            ctx_avg = ctx_masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        elif self.sentence_encoding=="range":
            B, C, E = ctx_vecs.shape
            indices = torch.arange(C, device=ctx_vecs.device).unsqueeze(0).expand(B, C)
            start = ranges[:, 0].unsqueeze(1) # (B, 1)
            end = ranges[:, 1].unsqueeze(1)   # (B, 1)

            # Dynamic range mask: (B,C)
            msk = (indices >= start) & (indices < end)
            mask_expanded = msk.unsqueeze(-1).float() # (B, C, 1)

            ctx_masked = ctx_vecs * mask_expanded
            ctx_sum = ctx_masked.sum(dim=1) # (B, E)
            ctx_avg = ctx_sum / torch.clamp(msk.sum(dim=1, keepdim=True).float(), min=1e-9)
        return ctx_avg
    def encode_lemmas(self, lemma_ids : list, input_ids_lemma, attention_mask_lemma):
        """
        Encodes the Sanskrit lemma.
        Depending from lemma encoding type, this function retrieves static embeddings,
        or encodes lemmas with a mini-transformer
        """
        if self.lemma_encoding=="static":
            return self.lemma_embeddings(lemma_ids)
        elif self.lemma_encoding=="transformer":
            return self.lemma_encoder(input_ids_lemma,attention_mask_lemma) # calls forward(...) of the encoder
        elif self.lemma_encoding=="none":
            return None # No lemma information.
    def encode_senses(self, sense_ixes, input_ids_def, attention_mask_def):
        if self.sense_encoding=="sensim":
            # Cached embeddings of the sensim model.
            return self.sense_projection(self.sense_embedding_matrix[sense_ixes]) # batch x model.maxNumDefs x self.sense_embedding_dim
        elif self.sense_encoding=="transformer":
            '''
            input_ids have dimensions batch x model.maxNumDefs x tokenizer.max_sen_length
            '''
            B, N, L = input_ids_def.shape
            o = self.definition_encoder[0]({
                'input_ids': input_ids_def.view(B * N, L),
                'attention_mask': attention_mask_def.view(B * N, L)
                })
            # Apply the pooling layer
            sentence_embedding = self.definition_encoder[1]({
                'token_embeddings': o["token_embeddings"], # o.last_hidden_state, # mpnet
                'attention_mask': attention_mask_def.view(B * N, L)
                })['sentence_embedding'] # B*N,D
            
            p = self.sense_projection2(sentence_embedding) # B*N, E
            return p.view(B, N, self.sense_embedding_dim) # B, N, D

    def forward(self, lemma_id, input_ids, attention_mask, 
                meaning_ids, 
                input_ids_lemma,
                attention_mask_lemma,
                input_ids_def,
                attention_mask_def,
                ranges,
                labels=None, soft_labels=None):
        lemmas = self.encode_lemmas(lemma_id,input_ids_lemma,attention_mask_lemma)  # (batch) --> (batch, emb_dim). Can return None!
        sen = self.encode_sentence(input_ids,ranges, attention_mask)  # transformer --> (batch, ctx_len, emb_dim)
        
        if self.architecture=="bilinear":
            d = self.sense_embedding_dim # sense embedding dimensionality
            projected = self.meaning_projection(sen)     # (B, d)
            W_bi = self.bilinear_projection(lemmas)         # (B, d*d)
            W_bi = W_bi.view(-1, d, d)                         # (B, d, d)
        else:
            if self.lemma_encoding=="none":
                projected = self.meaning_projection(sen)
            else:
                projected = self.meaning_projection(
                    torch.cat([lemmas, sen], dim=-1)  # (batch, emb_dim + transformer.hidden_size) > ...
                                                )  # (batch, sense_emb_dim)

        # Lookup sense embeddings
        # Mask: valid sense indices >= 0
        sense_mask = (meaning_ids >= 0)  # (batch, max_senses)
        sense_ixes = meaning_ids.clone()
        sense_ixes[sense_ixes < 0] = 0  # avoid indexing error

        # Gather the embeddings of the lexicographic definitions.
        sense_embs = self.encode_senses(sense_ixes,input_ids_def,attention_mask_def)  # (batch, max_senses, sense_emb_dim)

        # Compute dot product between @projected (sentence) and each LD embedding
        if self.architecture == "bilinear":
            #sen_proj, W = projected
            logits = torch.einsum('bd,bdk,bnk->bn', projected, W_bi, sense_embs)
            '''
            intermediate = torch.einsum('bd,bdk->bk', projected, W_bi)
            intermediate = torch.relu(intermediate)  # or tanh, gelu, etc.
            output = torch.einsum('bk,bnk->bn', intermediate, sense_embs)
            '''
        else:
            logits = torch.einsum('bd,bnd->bn', projected, sense_embs)

        # Mask padded senses
        logits = logits.masked_fill(~sense_mask, -1e4) # for fp16

        if labels is not None:
            nValidSenses = sense_mask.sum(dim=1)  # (batch,)
            ssm = (nValidSenses == 1) # single sense mask
            msm = (nValidSenses > 1)  # multi-sense mask
            losses = []
            if ssm.any():
                # For single-sense cases, use cosine similarity loss
                losses.append(1 - F.cosine_similarity(projected[ssm], sense_embs[ssm,0,:], dim=-1)) # n_single

            if msm.any():
                # For multi-sense cases: default cross-entropy
                if self.useSoftTargets:
                    # nyi/not tested
                    losses.append(-(soft_labels[msm] * F.log_softmax(logits[msm], dim=1)).sum(dim=1)) # n_multi
                else:
                    losses.append(F.cross_entropy(logits[msm], labels[msm], reduction='none'))  # n_multi
            loss = torch.cat(losses).mean()
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}
