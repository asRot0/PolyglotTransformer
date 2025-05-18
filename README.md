# PolyglotTransformer

> A multilingual Transformer-based Neural Machine Translation (NMT) system built with TensorFlow/Keras — modular, extendable, and production-ready.

**PolyglotTransformer** is a fully modular implementation of the Transformer architecture for sequence-to-sequence tasks, such as language translation. It supports language-specific notebooks and is designed for easy extensibility and experimentation.

- Built on TensorFlow 2 & Keras  
- Clean modular codebase under `models/`  
- Notebook-ready workflows for multilingual translation  
- Supports English to Bengali, German, and more  

```bash
PolyglotTransformer/
│
├── data/
├── models/
│   ├── __init__.py
│   ├── transformer_encoder.py       # Transformer Encoder Block
│   ├── transformer_decoder.py       # Transformer Decoder Block
│   ├── multi_head_attention.py      # Multi-Head Self-Attention
│   ├── feed_forward.py              # Position-wise Feed Forward Layer
│   ├── positional_encoding.py       # Sinusoidal Positional Encoding
│   ├── positional_embedding.py      # Token + Positional Embedding
│   └── transformer.py               # Full Encoder-Decoder Transformer
│
├── eng_to_ban.ipynb                 # English ➜ Bengali notebook
├── eng_to_japanese.ipynb            # English ➜ Japanese notebook
├── README.md                        
```

## Model Architecture

- Multi-head attention
- Position-wise feed-forward networks
- Layer normalization & residual connections
- Sinusoidal or learned positional embeddings
- Encoder–Decoder structure (as proposed in "Attention Is All You Need")


### 🗾 Translation Demo (English ➜ Japanese)

| 🌐 English Sentence                         | 🇯🇵 Japanese Translation                        |
|--------------------------------------------|-------------------------------------------------|
| Do you have any plans for today?           | 今日の予定はありますか？                       |
| I forgot to bring my umbrella.             | 傘を持ってくるのを忘れました。                  |
| Let's meet at the cafe at 3 p.m.           | 午後3時にカフェで会いましょう。                 |
| This movie was really interesting.         | この映画は本当に面白かったです。                |
| I want to visit Japan someday.             | いつか日本を訪れたいです。                      |


### 🌍 Translation Demo (English ➜ Bengali)

| 🌐 English Sentence                         | 🇧🇩 Bengali Translation                      |
|--------------------------------------------|---------------------------------------------|
| What are you doing?                        | তুমি কী করছো?                               |
| I love learning new languages.             | আমি নতুন ভাষা শিখতে ভালোবাসি।              |
| Can you help me, please?                   | তুমি কি আমাকে সাহায্য করতে পারো?           |
| This is my favorite book.                  | এটি আমার প্রিয় বই।                         |
| We need to leave early tomorrow morning.   | আমাদের আগামীকাল সকালেই রওনা দিতে হবে।     |

> 📝 *Translations generated using the trained Transformer model.*

## Extend to More Languages

Just copy an existing notebook like `eng_to_ban.ipynb` and change:
- The parallel corpus (`.txt`) file
- `source_vectorization` / `target_vectorization` setup
- Save as `eng_to_german.ipynb` or similar!

#### Recommended Dataset Source

You can download free English-to-X language sentence pairs from:
🔗 [https://www.manythings.org/anki/](https://www.manythings.org/anki/)

Example files:
- `eng-ger.txt` → English to German
- `eng-jpn.txt` → English to Japanese
- `eng-ita.txt` → English to Italian
- `eng-spa.txt` → English to Spanish

Just place the downloaded `.txt` file into your project and update the `load_data()` function accordingly!
