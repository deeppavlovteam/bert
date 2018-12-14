from run_classifier import *

from typing import List, Tuple, Callable

from flasgger import Swagger, swag_from
from flask import Flask, request, jsonify, redirect, Response
from flask_cors import CORS
import os

import sys, getopt


def interact(model: Callable[[list, list], list]) -> Tuple[Response, int]:
    if not request.is_json:
        log.error("request Content-Type header is not application/json")
        return jsonify({
            "error": "request Content-Type header is not application/json"
        }), 400

    data = request.get_json()
    text1 = data.get('text1')
    text2 = data.get('text2')
    if not isinstance(text1, list) or not isinstance(text2, list) or not len(text1) or not len(text1) == len(text2):
        return jsonify(
            {'error': f'expected two nonempty arrays of the same lengths, but got `{text1}` and `{text2}`'}), 400

    prediction = model(text1, text2)
    return jsonify(prediction), 200


def main(argv, host='0.0.0.0', port=5000):
    try:
        opts, args = getopt.getopt(argv, "hm:", ["model_dir="])
    except getopt.GetoptError:
        print('paraphraser_bert_server.py -m <model_dir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('paraphraser_bert_server.py -m <model_dir>')
            sys.exit()
        elif opt in ("-m", "--model_dir"):
            model_dir = arg

    max_seq_length = 128
    learning_rate = 2e-5
    do_lower_case = True
    train_batch_size = 32
    eval_batch_size = 8
    predict_batch_size = 8
    vocab_file =  os.path.join(model_dir, "vocab.txt")
    init_checkpoint = os.path.join(model_dir, 'model.ckpt-675')
    bert_config_file = os.path.join(model_dir, "bert_config.json")

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    if max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (max_seq_length, bert_config.max_position_embeddings))

    processor = MrpcProcessor()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=init_checkpoint,
        learning_rate=learning_rate,
        num_train_steps=None,
        num_warmup_steps=None,
        use_tpu=False,
        use_one_hot_embeddings=False)

    tpu_cluster_resolver = None
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=None,
        model_dir=model_dir,
        save_checkpoints_steps=1000,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=1000,
            num_shards=8,
            per_host_input_for_training=is_per_host))

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        predict_batch_size=predict_batch_size)


    def model(text1: list, text2: list) -> list:
        header = [['index', 'id1', 'id2', 'text1', 'text2']]
        examples = header + [['0', '1', '1', el[0], el[1]] for el in zip(text1, text2)]
        predict_examples = processor._create_examples(examples, "test")
        predict_file = os.path.join(model_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                max_seq_length, tokenizer,
                                                predict_file)
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=max_seq_length,
            is_training=False,
            drop_remainder=False)
        result = estimator.predict(input_fn=predict_input_fn)
        return [int(round(el[1])) for el in result]

    app = Flask(__name__)
    Swagger(app)
    CORS(app)

    endpoint_description = {
        'description': 'A model endpoint',
        'parameters': [
            {
                'name': 'data',
                'in': 'body',
                'required': 'true',
                'example': {
                    'text1': ['value'],
                    'text2': ['value']
                }
            }
        ],
        'responses': {
            "200": {
                "description": "A model response"
            }
        }
    }

    @app.route('/')
    def index():
        return redirect('/apidocs/')

    @app.route('/model', methods=['POST'])
    @swag_from(endpoint_description)
    def answer():
        return interact(model)

    app.run(host=host, port=port, threaded=False)


if __name__ == '__main__':
    main(sys.argv[1:])

