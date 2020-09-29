# Some useful scripts
## Text tokenization
BERT pre-training supposes that documents are splitted by empty line and one sentence per line.
We run tokenization before vocabulary building to control how texts are tokenized:
```bash
python tokenize_file.py path_to_text_file
```
As result, `..._tokenized.txt` file will be saved next to not tokenized file.

## Vocabulary building
We utilize YouTokenToMe to build subtokens vocabulary and transform it to BERT format.
```bash
python build_vocab.py path_to_tokenized_texts vocab_size vocab_coverage path_to_bert_vocab
```

New vocabulary will be save next to tokenized texts. BERT vocab is used to copy BERT special tokens to the new vocab.

## Re-assemble pre-trained checkpoint with new vocabulary
```bash
python reassemble_checkpoint.py target_model_path vocab_path base_model_path
```
`base_model` is used as a source, `target_model` is build with new `vocab`.

## Run pre-training
Follow original BERT pre-training instructions. To run pre-training run more efficiently, all texts should be pre-processed (build .tf_records). Multiple shards could be used in case if training set is large.