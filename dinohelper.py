
import torch
import torch.nn as nn

from encoder import encode, decode

import time


class DinoEmbedding(nn.Module):

    def __init__(self, vocab_size=29, dmodel=64):
        super().__init__()

        # E.Shape = [vocab_size, dmodel]
        self.E = nn.Parameter(torch.randn(vocab_size, dmodel))

    def forward(self, x):
        # Input: [B, seqlength]
        # Output: [B, seqlength, dmodel]

        # x.shape = [B, seqlength]
        # Each element of x is a token ID in [0, vocab_size-1]
        #
        # E[x] performs a row lookup:
        # [B, seqlength] -> [B, seqlength, dmodel]
        #
        # Output.shape = [B, seqlength, dmodel]
        
        return self.E[x]

class PositionEncoder(nn.Module):

    def __init__(self, seqlength=27, dmodel=64):
        super().__init__()

        self.seqlength = seqlength
        self.dmodel = dmodel

        # PE.shape = [seqlength, dmodel]
        PE = torch.zeros(seqlength, dmodel)

        for pos in range(seqlength):

            # i = 0, 2, 4, ... 62
            # Even dimensions use sin
            # Odd dimensions use cos
            for i in range(0, dmodel, 2):

                denominator = 10000 ** (i / dmodel)

                PE[pos, i] = torch.sin(
                    torch.tensor(pos / denominator)
                )

                PE[pos, i + 1] = torch.cos(
                    torch.tensor(pos / denominator)
                )

        # PE is part of the model, but is NOT trainable.
        self.register_buffer("PE", PE)


    def forward(self, X):

        # X.shape  = [B, seqlength, dmodel]
        # PE.shape = [   seqlength, dmodel]
        #
        # Decoder will perform:
        #
        # X + PE
        #
        # [B,27,64] + [27,64]
        #       -> [B,27,64]
        #
        # PE broadcasts across the batch dimension.

        return self.PE


class AttentionHead(nn.Module):

    def __init__(self, dmodel=64, headers=4):
        super().__init__()

        self.dmodel = dmodel
        self.dk = dmodel // headers
        self.dq = self.dk
        self.dv = self.dk

        # Wq.shape = [dmodel, dq] = [64,16]
        # Wk.shape = [dmodel, dk] = [64,16]
        # Wv.shape = [dmodel, dv] = [64,16]

        self.Wq = nn.Parameter(torch.randn(self.dmodel, self.dq))
        self.Wk = nn.Parameter(torch.randn(self.dmodel, self.dk))
        self.Wv = nn.Parameter(torch.randn(self.dmodel, self.dv))

    def forward(self, X):

        # X.shape = [B, seqlength, dmodel]
        #
        # [B,n,64] x [64,16] -> [B,n,16]

        Q = torch.matmul(X, self.Wq)
        K = torch.matmul(X, self.Wk)
        V = torch.matmul(X, self.Wv)

        # Q.shape = [B,n,16]
        # K.T     = [B,16,n]
        #
        # [B,n,16] x [B,16,n]
        #       -> [B,n,n]

        A = torch.matmul(Q, K.transpose(-2, -1))

        # Scaled dot-product attention
        A = A / (self.dk ** 0.5)

         # ----------------------------------------
        # CAUSAL MASK
        # ----------------------------------------

        n = X.shape[1]

        # mask.shape = [n,n]
        #
        # 1 0 0 0
        # 1 1 0 0
        # 1 1 1 0
        # 1 1 1 1

        mask = torch.tril(
            torch.ones(n, n, device=X.device)
        )

        # A.shape    = [B,n,n]
        # mask.shape = [  n,n]
        #
        # mask broadcasts across batch dimension.
        #
        # Future positions become -infinity.

        A = A.masked_fill(
            mask == 0,
            float('-inf')
        )

        # Softmax converts scores to attention weights.
        #
        # -inf -> probability 0
        #
        # A.shape remains [B,n,n]

        # A.shape remains [B,n,n]
        A = torch.softmax(A, dim=-1)

        # [B,n,n] x [B,n,16]
        #       -> [B,n,16]

        V_attn = torch.matmul(A, V)

        return V_attn
        


class MHA(nn.Module):

    def __init__(self, dmodel=64, headers = 4):
        
        super().__init__()

        self.attention1 = AttentionHead(dmodel, headers)
        self.attention2 = AttentionHead(dmodel, headers)
        self.attention3 = AttentionHead(dmodel, headers)
        self.attention4 = AttentionHead(dmodel, headers)

        self.Wo = nn.Parameter(torch.randn(dmodel, dmodel))
        self.bo = nn.Parameter(torch.randn(dmodel))
        

    def forward(self, X):

        # output of attention layers X1...H4 is [B, seqlength, dk=64/4 = 16]
        X1 = self.attention1(X)
        X2 = self.attention2(X)
        X3 = self.attention3(X)
        X4 = self.attention4(X)

        # concat headers X1,...X4 such that we get [B, seqlenght, dmodel (= dk * headers, 16 * 4 = 64)]
        X = torch.concat((X1, X2, X3, X4), dim=-1)

        # Wo has dimensions [dk * headers (or dmodel), dmodel]
        # Output dimentions [B, seqlength, dmodel]
        X = torch.matmul(X, self.Wo) + self.bo

        return X

class FFN(nn.Module):

    def __init__(self, dmodel=64):

        super().__init__()
        self.dmodel = dmodel

        self.dff = dmodel * 4

        self.W1 = nn.Parameter(torch.randn(self.dmodel, self.dff))
        self.b1 = nn.Parameter(torch.randn(self.dff))
        self.relu = nn.ReLU()

        self.W2 = nn.Parameter(torch.randn(self.dff, self.dmodel))
        self.b2 = nn.Parameter(torch.randn(self.dmodel))
        

    def forward(self, X):

        # X.shape = [B, seqlength, dmodel]
        # W1.shape = [dmodel, dff]
        # b1.shape = [dff]
        # X x W = [B, seqlenght, dff]
        X = torch.matmul(X, self.W1) + self.b1
        X = self.relu(X)
        X = torch.matmul(X, self.W2) + self.b2
        

        return X


class Decoder(nn.Module):

    def __init__(self, vocabsize=29, seqlength=27, dmodel = 64):
        super().__init__()

        self.seqlength = seqlength
        self.dmodel = dmodel
        self.vocabsize = vocabsize

        # Initiate the sub Networs
        self.dino_embedding = DinoEmbedding(vocabsize, dmodel)
        self.position_encoder = PositionEncoder(seqlength, dmodel)
        self.mha = MHA(dmodel=dmodel)
        self.norm1 = nn.LayerNorm(dmodel)
        self.ffn = FFN(dmodel=dmodel)
        self.norm2 = nn.LayerNorm(dmodel)
        self.linear = nn.Linear(dmodel, vocabsize)

    def forward(self, X):

        # Input Embedding
        X = self.dino_embedding(X)
        
        # Position Encoding
        X1 = self.position_encoder(X)
        X = X + X1

        # Multi Head Attention
        X2 = self.mha(X)
        X = X2 + X
        X = self.norm1(X)

        # Feed Forward Network
        X3 = self.ffn(X)
        X = X3 + X
        X = self.norm2(X)

        # Linear Projection to vocabulary
        # X has [B, seqlength, dmodel]
        # Output [B, seqlength, vocabsize]
        X = self.linear(X)

        return X
        

def generate_name(model, seed="Ganga", max_len=27):

    model.eval()

    # Training convention:
    # lowercase + START token
    name = "*" + seed.lower()

    print("Starting: " + name, end="", flush=True)

    with torch.no_grad():

        while len(name) < max_len:

            # ---------------------------------------
            # Prepare input
            # ---------------------------------------

            # Pad to seqlength = 27
            padded_name = name + "_" * (max_len - len(name))

            # Encode characters -> token IDs
            x = torch.tensor(
                encode(padded_name),
                dtype=torch.long
            )

            # Add batch dimension
            #
            # [27] -> [1,27]
            x = x.unsqueeze(0)

            # ---------------------------------------
            # Forward pass
            # ---------------------------------------

            # [1,27] -> [1,27,29]
            logits = model(x)

            # Position of last real character
            last_position = len(name) - 1

            # Get the 29 logits at that position
            #
            # [1,27,29]
            #       ↓
            # [29]
            next_logits = logits[0, last_position, :]

            # ---------------------------------------
            # Greedy decoding
            # ---------------------------------------

            # next_token = torch.argmax(next_logits).item()
            probabilities = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1).item()

            next_char = decode([next_token])

            # ---------------------------------------
            # END token?
            # ---------------------------------------

            if next_char == "$":
                # print(name + "$")
                break

            # Append predicted character
            name = name + next_char

            # print(name)
            # stream only the new character
            print(next_char, end="", flush=True)
            # time.sleep(0.12)
            

    print()
    # Remove START token before returning
    return name[1:]     

