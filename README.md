# 🦖 DinoGPT

A tiny transformer, built from scratch in PyTorch, that invents dinosaur names one letter at a time.

- **53,533 parameters**, one transformer block, 4 attention heads
- Trained on **1,535 real dinosaur names**
- Give it a seed (`golu`) and it finishes the name (`goluacesaurus`)

**🦖 Try it live: [dino-gpt.streamlit.app](https://dino-gpt.streamlit.app/)**

It's small enough to train on a laptop and simple enough that I know what every line does. This document explains how I built it, and why I made the choices I did.

## Why I built this

I really wanted to understand how transformers work, and the only way I trust that I understand something is to code it from the ground up.

So no `nn.Transformer`, no `nn.MultiheadAttention` and no pre-trained anything. Every weight matrix, every attention score and every mask is written out by hand, with the tensor shapes commented at each step. Generating dinosaur names was never really the goal. It's a small, fun task that made the real goal possible: build each part of a transformer myself, watch it learn, and know why every line is there.

## Inspiration

The idea comes from the *Dinosaurus Island* assignment in Andrew Ng's Deep Learning Specialization on Coursera (the Sequence Models course, on recurrent neural networks). There, a character-level **RNN** learns to generate dinosaur names.

DinoGPT is the same problem with a different machine. I swapped the RNN for a **transformer**, the architecture behind modern LLMs, introduced in [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017), and built every piece myself. Wherever this document says "the paper," it means that one. The task stays simple, so all the attention goes to the model.

The two read a name very differently. An RNN carries a hidden state forward one letter at a time, so every earlier letter reaches the next prediction only through that running summary. A transformer with a causal mask lets every position look straight back at every earlier letter through attention.

---

## The idea in one line

Give the model the start of a name, have it predict the next letter, add that letter, and repeat until it says the name is finished.

```
*  → a
*a → a
*aa → c
...
*aachenosaurus → $   (done)
```

That's the whole game. Everything below supports it.

---

## Build order

I didn't write the files in the order they appear. Here's the order I actually built things in, because it says a lot about the choices.

### 1. Data first

**Getting names.** Inspired by Dr. Chuck Severance (*Python for Everybody*), I scraped Wikipedia's list of dinosaur names into `dino_names.txt`, one name per line. The `fhand` / `fout` variable names are a direct tribute.

**Tokens (`encoder.py`).** The model works on characters, not words. My vocabulary has 29 tokens:

| Token | ID | Meaning |
|---|---|---|
| `a`–`z` | 0–25 | letters |
| `*` | 26 | start of a name |
| `$` | 27 | end of a name |
| `_` | 28 | padding |

`encode` turns a string into a list of integer IDs, and `decode` turns IDs back into a string. There's no one-hot encoding anywhere. The IDs go straight into an embedding lookup.

Why characters? Dinosaur names are built from repeating pieces like *-saurus*, *-raptor*, *-don* and *tyranno-*. A model that reads letters can pick up those pieces on its own.

**Training rows (`namereader.py`).** Every name becomes one row per letter: each prefix of the name, paired with the letter that comes next. The final row's answer is `$`. So 1,535 names become **19,910 training rows** in `dino_train.csv`.

The same prefix shows up with many different answers. `*a` is followed by dozens of different letters across the dataset. That isn't noise. It's how the model learns a *probability* for each next letter instead of a single fixed answer, and sampling from those probabilities is what lets it invent new names instead of copying old ones.

**Dataset and DataLoader.** `DinoDataset` pads each prefix with `_` to a fixed length of 27, encodes it, and returns `(x, y)`. In a batch of 32, `x` is `[32, 27]` and `y` is `[32]`.

### 2. The model, top-down

I wrote the model from the outside in.

**`Decoder` came first,** with the inner modules as empty placeholders. Writing the top-level `forward` first meant it read like the block diagram from *Attention Is All You Need*, and it set the contract each piece had to meet:

```
token IDs           [B, 27]
  → embedding       [B, 27, 64]
  + position code   [B, 27, 64]
  → multi-head attention, + residual, LayerNorm
  → feed-forward,         + residual, LayerNorm
  → linear to vocab [B, 27, 29]
```

That's also why `PositionEncoder.forward` returns its table rather than `X + PE`. `Decoder` does `X = X + X1`, so the position encoder only had to hand back something the right shape.

**Then `FFN`, the easiest piece.** Two linear layers with a ReLU in between: 64 → 256 → 64, the paper's ×4 ratio. It's also the biggest piece, about 33k of the 53k parameters.

**Then `MHA`, before `AttentionHead`.** Same top-down idea: writing the multi-head wrapper first decided what one head had to return (`[B, n, 16]`), so four of them concatenate back to `[B, n, 64]` before the output projection `Wo`.

#### Multi-head attention: the misconception that cost me the most

The key lines in `MHA.forward`:

```python
X1 = self.attention1(X)   # [B, n, 16]
X2 = self.attention2(X)   # [B, n, 16]
X3 = self.attention3(X)   # [B, n, 16]
X4 = self.attention4(X)   # [B, n, 16]

X = torch.concat((X1, X2, X3, X4), dim=-1)   # [B, n, 64]
```

**What I expected at first:** the *input* gets split between the heads, with `X[..., 0:16]` going to head 1, `X[..., 16:32]` to head 2, and so on.

**What actually happens:** **every head sees the full 64-dimension `X`.** The split is in each head's *output*. Each head has its own `Wq`, `Wk` and `Wv` of shape `64 × 16`, so it takes in all 64 features and projects them down to its own 16. The four 16-dimension outputs are then concatenated back to 64.

```
                 ┌─ Wq,Wk,Wv (64→16) ─ head 1 ─ [B,n,16] ─┐
X [B,n,64] ──────┼─ Wq,Wk,Wv (64→16) ─ head 2 ─ [B,n,16] ─┼─ concat ─ [B,n,64] ─ Wo ─ [B,n,64]
 (full X to all) ├─ Wq,Wk,Wv (64→16) ─ head 3 ─ [B,n,16] ─┤
                 └─ Wq,Wk,Wv (64→16) ─ head 4 ─ [B,n,16] ─┘
```

**Why splitting the input would be worse:**
- **Embedding dimensions don't have assigned meanings.** What the model knows about a letter is spread across all 64 numbers. A head that only saw dimensions 0–15 would be blind to three quarters of that information.
- **Full input lets each head choose what to look at.** Each head's projections learn *which mix* of all 64 features matters for its own job, such as tracking the previous letter or finding the `*` start token. With input slices, each head would be stuck with whatever happened to land in its slice.
- **The heads would be weaker.** Slicing the input would make each head's projections `16 × 16`, a quarter of the weights, and each head could only work with its own slice.

**Why the misconception is so easy to fall into.** Library code (and the paper's wording, "split into heads") *does* slice something, but it slices **after** projecting:

```python
Q = X @ Wq_big                          # one 64×64 matrix: full X in, [B, n, 64] out
Q = Q.view(B, n, 4, 16).transpose(1, 2) # [B, 4, n, 16], cut into 4 heads after projection
```

Since `Q[..., 0:16] == X @ Wq_big[:, 0:16]`, slicing the projected `Q` is exactly the same as giving head 1 its own `64 × 16` matrix applied to the full `X`, which is what my code does explicitly. My four separate heads are the readable version, and the reshape is the fast version of the same math.

**Why the output has to be 64 again.** The residual connection `X = MHA(X) + X` adds the attention output back to its input, so the shapes must match. That's what forces `dk × heads = dmodel` (16 × 4 = 64), and why I chose a `dmodel` that splits evenly.

**Four heads cost the same as one.** One head with 64-dimension Q, K and V uses 3 × 64 × 64 = 12,288 parameters. Four heads with 16 dimensions each use 4 × 3 × 64 × 16 = 12,288. Same price, but four separate `n × n` attention maps instead of one, each free to learn a different pattern. `Wo` then learns how to combine them.

**Then `AttentionHead`.** Q, K and V projections, scaled dot-product, a **causal mask**, softmax, then multiply by V. The mask is the part that makes this a *language model*, so it gets its own section below.

#### The causal mask: why it's there and what it does

**What attention does without a mask.** Attention lets every position look at every other position. For each character, the head scores how relevant each character in the sequence is (`Q · Kᵀ`), turns the scores into weights with softmax, and takes a weighted mix of their values (`V`). With no mask, "every position" includes the ones *after* it.

**Why that's a problem for a language model.** The model's whole job is to predict the next letter from the letters so far. If position 3 can see position 4, it isn't predicting the next letter. It's reading it. A model trained like that would score well in training and fail at generation, because when it's actually writing a name, the next letter doesn't exist yet. The mask makes training match generation: **each position only ever sees the past.**

**How it works in code.** For a sequence of length `n`:

```python
mask = torch.tril(torch.ones(n, n))          # 1 = allowed to look, 0 = future
A = A.masked_fill(mask == 0, float('-inf'))  # future scores → -inf
A = torch.softmax(A, dim=-1)                 # e^(-inf) = 0 → zero weight on the future
```

For the input `*rex`, the mask looks like this (rows = the position doing the looking, columns = what it can see):

|        | `*` | `r` | `e` | `x` |
|--------|:---:|:---:|:---:|:---:|
| **`*`** | ✅ | ❌ | ❌ | ❌ |
| **`r`** | ✅ | ✅ | ❌ | ❌ |
| **`e`** | ✅ | ✅ | ✅ | ❌ |
| **`x`** | ✅ | ✅ | ✅ | ✅ |

Some details that matter:
- I set the masked scores to `-inf` **before** softmax, not to 0 after it. That way the weights that remain still add up to 1.
- The diagonal is always allowed, so every row has at least one position it can see. If a row were all `-inf`, softmax would return NaN.
- The mask is `[n, n]` while the scores are `[B, n, n]`. PyTorch applies the same mask to every name in the batch automatically (broadcasting).

**What it does for DinoGPT specifically.**
1. **Padding takes care of itself.** Inputs are padded with `_` to length 27. The last real character can't look ahead at the padding, and the loss only reads that position. So padding never affects a prediction, and I didn't need a separate padding mask.
2. **Every position becomes a valid predictor.** Because position *i* sees only characters `0..i`, its output is exactly the prediction for the prefix ending there. Feed in `*rex` and position 0 predicts what follows `*`, position 1 what follows `*r`, and so on, all from one forward pass.

**So was the mask actually required here? Honestly, no.** In the way I train DinoGPT, the model would work without it:

- **The answer is never in the input.** Each row is a prefix (`*aachen`) with its target (`o`) in a separate column. There's no future letter in the sequence to peek at, so there's nothing to cheat with.
- **The padding adds nothing.** Without the mask, the last real character could also attend to the `_` padding after it. But the amount of padding depends only on the prefix length, which the position encoding already gives it. Generation pads the same way, so training and generation still match.
- **One layer, one position read.** The last real position already sees every real character *with* the mask on. So removing the mask changes just one thing: some attention can go to padding. At best that's harmless. At worst it pulls a little attention away from the real letters.
- **Shuffling is unrelated.** Shuffling changes the order of the rows. The mask works *inside* a row. Each row goes through the model on its own either way.

So in this version, the causal mask is really doing a padding mask's job, and even that is optional. I kept it because it's part of the GPT design I set out to learn, and because it becomes **essential** for the next step (see *What I'd change next*). Training on whole names at once puts the answer for position *i* right there at position *i+1*, and without the mask the model would simply copy it.

*Experiment to try:* comment out the `masked_fill` line, retrain for 50 epochs, and compare the final loss. I'd expect it to be about the same (around 1.55), possibly slightly worse.

**Encoder or decoder?** I think of the block as encoder-style: self-attention plus feed-forward, with no cross-attention. Adding the causal mask is exactly what turns an encoder block into a GPT-style decoder block. That's why the class is called `Decoder`.

**`PositionEncoder` last,** before the first forward pass. Attention by itself has no sense of order, so position has to be added in.

**Everything is built by hand** from raw `nn.Parameter` matrices and `torch.matmul`. Only `LayerNorm` and the final `Linear` come from PyTorch. The goal was to learn, not to be quick.

### 3. Forward pass, training and generation

Once the pieces fit, these parts were simple.

**Training.** The model outputs predictions at all 27 positions, but each row only has one answer: the letter after the last real character. So I find that position (count the non-padding tokens, minus one), take its 29 logits, and compute cross-entropy against the target. Adam, learning rate 1e-3, batch size 32, 50 epochs. Loss goes from about 3.6 before training (random guessing over 29 tokens would be ln 29 ≈ 3.37) down to about 1.55 after 50 epochs.

**Generation.** Start with `*` plus whatever seed the user types, run the model, read the prediction at the last real character, **sample** the next letter from the probabilities, add it, and repeat until the model outputs `$`.

**It knows when a name is finished.** When I seed it with a complete real name (a whole dinosaur name, not just a start), it closes right away with `$`. Nobody hard-coded that. Endings like *-saurus*, *-raptor* and *-don* are followed by `$` hundreds of times in the training rows, so the model learned that a name ending that way is done.

### 4. From notebook to web app

Once the model worked in the notebook, I wanted people to use it without opening Jupyter. That took two steps.

**`dinohelper.py`: an independent loader.** I moved the model classes (`DinoEmbedding`, `PositionEncoder`, `AttentionHead`, `MHA`, `FFN`, `Decoder`) and `generate_name` out of the notebook into a plain Python module. Any script can now rebuild the model and load the trained weights with one import, without needing the notebook or retraining:

```python
from dinohelper import Decoder, generate_name

model = Decoder()
model.load_state_dict(torch.load("dino_gpt.pth"))
model.eval()
```

The weights file only stores numbers. The code that knows what those numbers *mean* (the architecture) has to come from somewhere, and `dinohelper.py` is that somewhere.

**`feedforward.py`: the Streamlit app.** It imports from `dinohelper.py`, loads `dino_gpt.pth` once, and puts a text box and a 🦖 **Generate Name** button on a web page. The name comes back in bold in a green box:

> Our latest Dinosaur is named **SHISHIROSURK**. Roar!

> Our latest Dinosaur is named **FIGMAOCAR**. Roar!

This is the same `generate_name` loop as in the notebook. Only the front end changed.

---

## Choices I made, and why

| Choice | Why |
|---|---|
| **Character-level tokens** | Names are made of reusable letter-pieces, and a 29-token vocabulary keeps the model tiny. |
| **`*` / `$` markers** | The model needs to know where a name starts, and it needs to be able to say "I'm done." |
| **Padding to 27** | Fixed-length inputs stack into one tensor per batch. |
| **`dmodel = 64`** | It splits cleanly into 4 heads × 16 dimensions, and at this size memory wasn't a concern. |
| **Sinusoidal positions** | At first I wanted a learned position network, but I chose to stay true to the original paper. (GPT-2 uses learned positions, and with a fixed length of 27 either one would work.) |
| **Causal mask** | Each position predicts the next letter using only what came before, the same setup GPT uses. It isn't strictly required by my prefix-per-row training, but it's needed for whole-name training. See *The causal mask* above. |
| **Read only the last real position** | Each training row asks exactly one question: what comes after this prefix? |
| **Sampling instead of argmax** | Always picking the top letter gives the same name every time. Sampling makes every run a new dinosaur. |
| **Separate `dinohelper.py`** | Gives the app (and anything else) an independent way to load the model, without depending on the notebook. The notebook is for training; the module is for using. |
| **Streamlit for the app** | A working web page in about 20 lines of Python, with no HTML or JavaScript, and free hosting on Streamlit Community Cloud straight from GitHub. |
| **`time.sleep(0.12)` while printing** | Added later. The model is so fast the name appeared all at once, so the delay streams it letter by letter, like a chatbot "thinking". Built for my six-year-old's demo, along with the 🦖 MAKE A DINO! button. (It held his attention for about two minutes.) |

---

## Lessons learned

- **Shapes will eat an afternoon.** Mine went to getting the dimensions right through attention: `K.transpose(-2, -1)` rather than `.T` (which flips every dimension on a 3D tensor), and getting a `[n, n]` mask to apply across the batch. The shape comments throughout the code are scars from that afternoon.
- **Heads split the output, not the input.** I first assumed each attention head got a 16-dimension slice of `X`. In fact each head sees all of `X` and projects it down to 16. See *Multi-head attention: the misconception that cost me the most*.
- **Test stubs before filling them in.** Next time, each placeholder module will return a correctly shaped tensor, and `Decoder` will check its output shape, so one batch through the model shows shape bugs in seconds.
- **Check the scraped data.** Ten lines in `dino_names.txt` are two names stuck together (e.g. `LisboasaurusLiubangosaurus`), probably from list items that had no line break between them when I scraped them. They're the reason the max length is 27 instead of 24, and they teach the model to keep going after an ending like *-saurus* instead of ending the name.

---

## What I'd change next

- **Start the weights smaller.** Every weight matrix starts as `torch.randn` (values around ±1). That makes the attention scores very large at the start, so softmax puts nearly all the weight on one position and gradients are tiny. Scaling by about `1/√(input size)`, as `nn.Linear` does, and starting biases at zero should make training noticeably better.
- **Train on whole names at once.** Thanks to the causal mask, one forward pass over `*aachenosaurus` already produces all 14 next-letter predictions. Training on 1,535 full names with targets shifted by one (and `ignore_index` for padding) gives the same training signal as 19,910 prefix rows, at about 13× less compute. That's how GPT is actually trained.
- **Fix the 10 merged names** and work out the max length from the data instead of hard-coding 27 in three places.
- **Shuffle once.** The DataLoader's `shuffle=True` already does it; the pandas shuffle is redundant.
- Multiply the embeddings by √dmodel before adding positions, as the paper does.
- **Harden the app before it goes public.** The tokenizer only knows a–z, so a seed with spaces, digits or punctuation (`T-Rex 2`) will crash it. Keep only letters and cap the length. Also cache the model with `@st.cache_resource` so it isn't reloaded on every click, and load it with `map_location="cpu"` for servers without a GPU.
- **Add a temperature setting.** Dividing the logits by a temperature before the softmax would let users choose between safe names and wilder ones. Long seeds currently give the same answer every time: `varunapithecu` always becomes **VARUNAPITHECUS**, because the model is nearly certain that `cu` → `s` → end.
- **Also publish on Hugging Face,** as a Gradio Space, next to the Streamlit version.

---

## What this taught me

Looking back, this exercise taught me **how a base LLM is trained**. DinoGPT runs the same loop as pretraining a large model: tokenize text, predict the next token, compute cross-entropy against the real next token, backpropagate, repeat. Everything else is scale:

| | DinoGPT | A base LLM |
|---|---|---|
| Tokens | 29 characters | ~50k–200k subword tokens (BPE) |
| Data | 1,535 names | trillions of tokens |
| Depth | 1 block, 4 heads | dozens of blocks, dozens of heads |
| Context | 27 characters | thousands to millions of tokens |
| Objective | next-token cross-entropy | **the same** |

The loop doesn't change, only the size.

What turns a base model into an assistant sits **on top** of this loop:

- **SFT (supervised fine-tuning):** the *same* next-token loss, run on curated prompt → response examples. Typically only the response tokens count toward the loss. That's the same idea I used when reading only the last real position: compute loss only where you want the model to learn.
- **RLHF (and alternatives like DPO):** this is where the objective itself changes. Instead of "predict the real next token," the model is pushed toward responses that people, or a reward model trained on their preferences, rate higher.
- **Other fine-tuning** (including parameter-efficient methods like LoRA): continuing training on narrower data, often updating only a small set of added weights.

So pretraining is the foundation, SFT is the same machinery aimed at better examples, and RLHF is where a genuinely new idea comes in. Building the foundation by hand, one tensor shape at a time, is what made the rest make sense to me.

---

## Files

| File | What it does |
|---|---|
| `dino_names.txt` | 1,535 dinosaur names scraped from Wikipedia |
| `namereader.py` | Turns names into prefix → next-letter rows |
| `dino_train.csv` | 19,910 training rows |
| `encoder.py` | Character tokenizer (29 tokens) |
| `word_converter.py` | Quick tokenizer and padding test |
| `dinogpt.ipynb` | Model, training, generation, and the original 🦖 button |
| `dino_gpt.pth` | Trained weights |
| `dinohelper.py` | The model classes and `generate_name`, as an importable module |
| `feedforward.py` | Streamlit web app: type a seed, get a dinosaur |
| `requirements.txt` | Packages the app needs (`torch`, `streamlit`) |

## Running the app

The app is live at **[dino-gpt.streamlit.app](https://dino-gpt.streamlit.app/)**, hosted free on Streamlit Community Cloud, which redeploys automatically on every push to `main`. To run it locally:

```bash
pip install -r requirements.txt
streamlit run feedforward.py
```

Then open the address it prints (usually `http://localhost:8501`), type the start of a name, or nothing at all, and press **Generate Name**.

## Parameter count

| Part | Parameters |
|---|---|
| Token embedding (29 × 64) | 1,856 |
| 4 attention heads (Q, K, V: 64 × 16 each) | 12,288 |
| Output projection `Wo` + bias | 4,160 |
| Feed-forward (64 → 256 → 64) | 33,088 |
| 2 × LayerNorm | 256 |
| Output layer (64 → 29) | 1,885 |
| **Total** | **53,533** |

The sinusoidal position table is a buffer, not a parameter, so it isn't trained or counted.

---

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762). Advances in Neural Information Processing Systems 30. This is the source of the transformer block, scaled dot-product attention, multi-head attention and sinusoidal position encoding.
- Andrew Ng, [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning), Coursera, Course 5: Sequence Models. The *Dinosaurus Island* character-level RNN assignment that inspired this project.
- Charles Severance, [*Python for Everybody*](https://www.py4e.com/). The inspiration for scraping and file handling.
