from pathlib import Path
import sys
from shutil import copyfile

import tensorflow as tf
import numpy as np
from tqdm import tqdm as tqdm

import bert_dp.modeling as modeling
from bert_dp.modeling import BertConfig, BertModel
from bert_dp.tokenization import FullTokenizer


class FullTokenizerST(FullTokenizer):
    """Same as original FullTokenizer, but can tokenize single token.
    """
    def __init__(self, vocab_file, do_lower_case=True):
        super().__init__(vocab_file=vocab_file, do_lower_case=do_lower_case)

    def tokenize(self, text, single_token=False):
        split_tokens = []
        if single_token:
            for sub_token in self.wordpiece_tokenizer.tokenize(text):
                split_tokens.append(sub_token)
        else:
            for token in self.basic_tokenizer.tokenize(text):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
        return split_tokens


# get_next_sentence_output, get_masked_lm_output, gather_indexes
# are taken from run_pretraining.py
def get_next_sentence_output(bert_config, input_tensor, labels):
    """Get loss and log probs for the next sentence prediction."""

    # Simple binary classification. Note that 0 is "next sentence" and 1 is
    # "random sentence". This weight matrix is not used after pre-training.
    with tf.variable_scope("cls/seq_relationship"):
        output_weights = tf.get_variable(
            "output_weights",
            shape=[2, bert_config.hidden_size],
            initializer=modeling.create_initializer(bert_config.initializer_range))
        output_bias = tf.get_variable(
            "output_bias", shape=[2], initializer=tf.zeros_initializer())

        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        labels = tf.reshape(labels, [-1])
        one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, log_probs)


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[bert_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator
    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


# set up
# TARGET_MODEL_NAME
# LOWERCASE
# BPE_PATH -- path to new vocab
# BASE_MODEL_NAME -- embedding matrix from base model would be reassembled
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python reassemble_checkpoint.py target_model_path vocab_path base_model_path')
        exit(1)

    LOWERCASE = False
    print(f'Warning, lowercase is set to {LOWERCASE}!')
    TARGET_MODEL_DIR = Path(sys.argv[1])
    BPE_PATH = sys.argv[2]
    BASE_MODEL_DIR = Path(sys.argv[3])

    BASE_MODEL_CKPT_PATH = str(BASE_MODEL_DIR / 'bert_model.ckpt')
    BASE_VOCAB_PATH = str(BASE_MODEL_DIR / 'vocab.txt')
    BASE_CONFIG_PATH = str(BASE_MODEL_DIR / 'bert_config.json')

    TARGET_MODEL_CKPT_PATH = str(TARGET_MODEL_DIR / 'bert_model.ckpt')
    TARGET_VOCAB_PATH = str(TARGET_MODEL_DIR / 'vocab.txt')
    TARGET_CONFIG_PATH = str(TARGET_MODEL_DIR / 'bert_config.json')

    max_seq_length = 512
    tf.reset_default_graph()

    bert_config = BertConfig.from_json_file(BASE_CONFIG_PATH)
    is_training = False
    use_one_hot_embeddings = False

    input_ids_ph = tf.placeholder(shape=(None, None), dtype=tf.int32)
    input_masks_ph = tf.placeholder(shape=(None, None), dtype=tf.int32)
    segment_ids_ph = tf.placeholder(shape=(None, None), dtype=tf.int32)
    masked_lm_positions_ph = tf.placeholder(shape=(None, None), dtype=tf.int32)
    labels = tf.placeholder(tf.int64, shape=[None, 1])
    label_ids = tf.placeholder(tf.int64, shape=[None, None])
    label_weights = tf.placeholder(tf.float32, shape=[None, None])

    model = BertModel(config=bert_config,
                      is_training=is_training,
                      input_ids=input_ids_ph,
                      input_mask=input_masks_ph,
                      token_type_ids=segment_ids_ph,
                      use_one_hot_embeddings=use_one_hot_embeddings)

    ns_prob = get_next_sentence_output(bert_config,
                                       model.get_pooled_output(),
                                       labels)

    masked_lm_probs = get_masked_lm_output(bert_config,
                                           model.get_sequence_output(),
                                           model.get_embedding_table(),
                                           masked_lm_positions_ph,
                                           label_ids,
                                           label_weights)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    sess = tf.Session(config=sess_config)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, BASE_MODEL_CKPT_PATH)

    # Get old embeddings matrix
    old_emb_mat = sess.run(model.embedding_table)

    # Get new BERT vocab transformed from yttm without BERT special tokens
    subword_vocab = [line.strip() for line in open(BPE_PATH, 'r', encoding='utf8')
                     if not ('[' in line and ']' in line)]

    # BERT vocab reading func
    def get_vocab(path):
        vocab = []
        with open(path, encoding='utf8') as f:
            for n, line in enumerate(f):
                if len(line) > 1:
                    token = line[:-1]
                    vocab.append(token)
        return vocab

    old_tokenizer = FullTokenizerST(vocab_file=str(BASE_VOCAB_PATH), do_lower_case=LOWERCASE)
    old_vocab = get_vocab(BASE_VOCAB_PATH)

    new_vocab = subword_vocab

    # extend old embeddings and old vocab with new tokens from new vocab
    # discard old tokens that do not fit into vocab_size
    new_emb_mat = np.zeros_like(old_emb_mat)

    index_new_vocab = 0
    count_leaved = 0
    count_en = 0
    new_vocab_set = set(new_vocab)

    # which tokens from new_vocab already added
    mask = np.zeros(len(new_vocab))

    vocab_reassembled = []
    count_chars = 0
    all_new_tokens_added = False

    for n, token in tqdm(enumerate(old_vocab)):
        if '—' in token:
            print(n)

        # copy all special tokens
        if n < 106:  # number of special tokens in BERT
            new_emb_mat[n] = old_emb_mat[n]
            vocab_reassembled.append(token)
            continue

        # if token from old_vocab is already in new_vocab we mark it as already added
        if token in new_vocab_set:
            if mask[new_vocab.index(token)]:
                # duplicates?
                pass
            else:
                if token == '!':
                    print(n)
                mask[new_vocab.index(token)] = 1
                token_ind = old_tokenizer.convert_tokens_to_ids([token])[0]
                new_emb_mat[n] = old_emb_mat[token_ind]
                count_leaved += 1
                vocab_reassembled.append(token)
                continue

        # adding new tokens
        if all_new_tokens_added:
            continue
        while index_new_vocab < len(mask) and mask[index_new_vocab]:
            index_new_vocab += 1
        if index_new_vocab == len(mask):
            all_new_tokens_added = True
            print('all tokens from new_vocab added, skipping...')
            continue
        token_to_add = new_vocab[index_new_vocab]
        tokens_to_average = old_tokenizer.tokenize(token_to_add, single_token=True)
        if any('[UNK]' in tok for tok in tokens_to_average):
            # try to tokenize lowercased
            tokens_to_average_l = old_tokenizer.tokenize(token_to_add.lower(), single_token=True)
            if any('[UNK]' in tok for tok in tokens_to_average_l):
                print(token_to_add, tokens_to_average)
        token_ids = old_tokenizer.convert_tokens_to_ids(tokens_to_average)
        new_emb_mat[n] = old_emb_mat[token_ids].mean(axis=0)
        vocab_reassembled.append(token_to_add)

        mask[index_new_vocab] = 1
        index_new_vocab += 1

    print('Old emb mean/std:', old_emb_mat.mean(), old_emb_mat.std())
    print('New emb mean/std', new_emb_mat.mean(), new_emb_mat.std())

    if len(vocab_reassembled) != len(old_vocab):
        print(f'Warning! len(new_vocab) != len(old_vocab)\n{len(vocab_reassembled)} {len(old_vocab)}')
        print('Embedding matrix shape will be changed!')

    if len(vocab_reassembled) != len(old_vocab):
        new_emb_mat = new_emb_mat[:len(vocab_reassembled), :]
        bias = tf.train.load_variable(BASE_MODEL_CKPT_PATH, 'cls/predictions/output_bias')
        print('Updating bias:')
        print('old:', bias.shape, np.mean(bias), np.std(bias))
        with tf.Session(config=sess_config) as sess:
            new_bias = sess.run(tf.truncated_normal(shape=(len(vocab_reassembled),),
                                                    mean=np.mean(bias), stddev=np.std(bias)))
        print('new:', new_bias.shape, np.mean(new_bias), np.std(new_bias))

    # assing new emb matrix
    if len(vocab_reassembled) == len(old_vocab):
        import numpy as np
        assign_op = tf.assign(model.embedding_table, new_emb_mat)
        sess.run(assign_op)
        sess.run(tf.global_variables_initializer())
        print(sess.run(tf.global_variables()[2]))
        print(sess.run(model.embedding_table))
        saver.save(sess, TARGET_MODEL_CKPT_PATH, write_meta_graph=False)

    # if you want to resize emb matrix
    if len(vocab_reassembled) != len(old_vocab):
        print('resizing emb matrix:')
        tf.reset_default_graph()
        with tf.Session(config=sess_config) as sess:
            for var_name, _ in tf.train.list_variables(BASE_MODEL_CKPT_PATH):
                var = tf.train.load_variable(BASE_MODEL_CKPT_PATH, var_name)
                new_name = var_name
                if var_name == 'bert/embeddings/word_embeddings':
                    var = new_emb_mat
                    print(f'set new val to {var_name}')
                if var_name == 'cls/predictions/output_bias':
                    var = new_bias
                    print(f'set new val to {var_name}')
                var = tf.Variable(var, name=new_name)
            # Save the variables
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.save(sess, TARGET_MODEL_CKPT_PATH, write_meta_graph=False)

    copyfile(BASE_CONFIG_PATH, TARGET_CONFIG_PATH)
    with open(TARGET_VOCAB_PATH, 'w', encoding='utf8') as f:
        for token in vocab_reassembled:
            f.write(token + '\n')
    print('DONE')
