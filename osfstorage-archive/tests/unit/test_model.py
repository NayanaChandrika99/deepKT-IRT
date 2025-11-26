from typing import Dict, List, Union

import numpy as np
import pytest
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from edm2023 import model, preprocessing


@pytest.fixture
def item_features():
    item_bert_embeddings = tf.constant(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        dtype=tf.float32,
    )

    multiple_choice_items = {0, 2}
    item_success_rate = tf.constant([0.5, 0.6, 0.7])

    return model.ItemFeatures(
        unit_test_item_size=3,
        multiple_choice_items=multiple_choice_items,
        item_success_rate=item_success_rate,
        item_bert_embeddings=item_bert_embeddings,
    )


@pytest.fixture
def in_unit_item_size():
    return 1000


@pytest.fixture
def wide_and_deep_irt_config():
    return {
        "hidden_size": 8,
        "dropout": 0.5,
        "unit_test_item_size": 100,
    }


@pytest.fixture
def deep_features() -> List[str]:
    return [
        "history_recency",
        "history_actions",
        "history_item_response_time_quantile_10",
        "history_item_success_rates",
        "history_item_bert_embeddings",
        "history_missing",
    ]


@pytest.fixture
def categorical_feature_embedding_layers(
    deep_features,
) -> Dict[str, tf.keras.layers.Layer]:
    res = {}

    # the output dimensions are arbitrarily configured below
    if "history_recency" in deep_features:
        res["history_recency"] = tf.keras.layers.Embedding(
            10_000, 1
        )  # 10_000 is just an arbitrary high number
    if "history_actions" in deep_features:
        res["history_actions"] = tf.keras.layers.Embedding(
            12, 1
        )  # in the paper, only top 10 actions are included, the extra two are for padding, and other
    if "history_item_response_time_quantile_10" in deep_features:
        res["history_item_response_time_quantile_10"] = tf.keras.layers.Embedding(11, 8)
    if "history_missing" in deep_features:
        res["history_missing"] = tf.keras.layers.Embedding(2, 8)

    return res


@pytest.fixture
def wide_and_deep_irt(
    wide_and_deep_irt_config,
    deep_features: List[str],
    categorical_feature_embedding_layers: Dict[str, tf.keras.layers.Layer],
):
    wdirt = model.WideAndDeepIRT(**wide_and_deep_irt_config)
    for feature in deep_features:
        wdirt.register_feature(feature)
    for feature, embedding_layer in categorical_feature_embedding_layers.items():
        wdirt.register_categorical_feature_embedding_layer(feature, embedding_layer)
    return wdirt


@pytest.fixture
def random_dataset(
    in_unit_item_size, wide_and_deep_irt_config, deep_features
) -> tf.data.Dataset:
    """Random dataset for training"""
    unit_test_item_size = wide_and_deep_irt_config["unit_test_item_size"]
    num_obs = 1000
    seq_len = 200

    item_bert_embeddings = tf.random.uniform(
        shape=[in_unit_item_size, 32], minval=-1.0, maxval=1.0, dtype=tf.float32
    )

    multiple_choice_items = np.random.randint(
        low=0, high=2, size=(wide_and_deep_irt_config["unit_test_item_size"],)
    ).astype(dtype="float32")

    item_success_rate = tf.random.uniform(
        shape=[in_unit_item_size], minval=0.0, maxval=1.0, dtype=tf.float32
    )

    item_bert_embeddings_lookup = preprocessing.Lookup(item_bert_embeddings)
    multiple_choice_items_lookup = preprocessing.Lookup(multiple_choice_items)
    item_success_rate_lookup = preprocessing.Lookup(item_success_rate)

    def process_data(
        inputs: Dict[str, Union[np.ndarray, tf.Tensor]],
        requested_deep_features: List[str] = deep_features,
    ):
        """Process data"""
        # wide features
        inputs["future_item_is_multiple_choice"] = multiple_choice_items_lookup(
            inputs["future_item"]
        )
        inputs["mask"] = tf.cast(tf.equal(inputs["in_unit_items"], 0), dtype="float32")

        # deep features
        if "history_item_success_rates" in requested_deep_features:
            inputs["history_item_success_rates"] = tf.expand_dims(
                item_success_rate_lookup(inputs["in_unit_items"]), axis=-1
            )  # output shape = [batch_size, seq_len, 1]

        if "history_item_bert_embeddings" in requested_deep_features:
            inputs["history_item_bert_embeddings"] = item_bert_embeddings_lookup(
                inputs["in_unit_items"]
            )  # output shape  = [batch_size, seq_len, hidden_size]

        # remove useless features
        inputs.pop("in_unit_items")

        # generate y
        numerical_features = [
            "history_item_success_rates",
            "history_item_bert_embeddings",
        ]
        numerical_feature_vals = tf.concat(
            [inputs[deep_feature] for deep_feature in numerical_features], axis=-1
        )

        prob = tf.nn.sigmoid(
            tf.reduce_sum(
                tf.reduce_sum(numerical_feature_vals, axis=-1, keepdims=False),
                axis=-1,
                keepdims=False,
            )
        )
        y = tf.cast(tf.greater(prob, 0.5), dtype="int32")

        return inputs, y

    x = {
        # wide features
        "future_item": np.random.randint(
            low=1, high=unit_test_item_size, size=(num_obs, 1)
        ),
        # intermediate features, will be removed in the `preprocess_data` function
        "in_unit_items": np.random.randint(
            low=1, high=in_unit_item_size, size=(num_obs, seq_len)
        ),
        # deep features
        "history_recency": np.random.randint(
            low=1, high=seq_len, size=(num_obs, seq_len)
        ),
        "history_missing": np.random.randint(low=0, high=2, size=(num_obs, seq_len)),
        "history_actions": np.random.randint(
            low=1,
            high=11,
            size=(num_obs, seq_len),
        ),
        "history_item_response_time_quantile_10": np.random.randint(
            low=0,
            high=11,
            size=(num_obs, seq_len),
        ),
    }

    dataset = tf.data.Dataset.from_tensor_slices(x)
    dataset = dataset.map(process_data)
    dataset = dataset.batch(128)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


@pytest.mark.parametrize(
    "action_len, action_size, hidden_size, seq_len, inputs",
    [
        (
            4,
            8,
            4,
            4,
            {
                "actions": [
                    [
                        [0, 1, 2, 3],
                        [0, 0, 4, 3],
                        [0, 1, 7, 5],
                        [0, 2, 3, 5],
                    ]
                ]
            },
        )
    ],
)
def test_model_lstm_encoding_layer(
    action_len, action_size, hidden_size, seq_len, inputs
):
    lstm_encoding_layer = model.LSTMActionEncodingLayer(
        feature_name="actions",
        action_len=action_len,
        action_size=action_size,
        hidden_size=hidden_size,
        seq_len=seq_len,
    )

    result = lstm_encoding_layer(inputs)

    # the layer is initialized with random weights, so only need to check the output shape
    input_batch_size = len(list(inputs.values())[0])
    assert result.numpy().shape == (input_batch_size, seq_len, hidden_size)


def test_model_wide_and_irt_forward(
    wide_and_deep_irt: model.WideAndDeepIRT, random_dataset: tf.data.Dataset
):
    # random_dataset is a generator
    for x, y in random_dataset:
        break
    try:
        wide_and_deep_irt(x)
    except Exception as e:
        assert False, str(e)


def test_model_wide_and_irt_train(
    wide_and_deep_irt: model.WideAndDeepIRT, random_dataset: tf.data.Dataset
):
    wide_and_deep_irt.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.05),
        loss="binary_crossentropy",
    )

    try:
        wide_and_deep_irt.fit(random_dataset, verbose=1, epochs=50)

        y_pred = wide_and_deep_irt.predict(random_dataset)
        y_pred = np.concatenate(y_pred, axis=0)

        y_true = []
        for x, y in random_dataset:
            y_true.append(y)
        y_true = np.concatenate(y_true)

        assert roc_auc_score(y_true, y_pred) > 0.95

    except Exception as e:
        assert False, str(e)
