from typing import Dict, Optional

import numpy as np
import tensorflow as tf

_NEG_INF_FP32 = -1e9


class LSTMActionEncodingLayer(tf.keras.layers.Layer):
    """LSTM encoding layer for actions"""

    def __init__(
        self,
        feature_name: str,
        action_len: int,
        action_size: int,
        hidden_size: int,
        seq_len: int,
        **kwargs,
    ):
        self.action_len = action_len
        self.feature_name = feature_name
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.action_size = action_size
        self.dense_embedding = tf.keras.layers.Embedding(action_size, hidden_size)
        self.lstm = tf.keras.layers.LSTM(hidden_size)
        super(LSTMActionEncodingLayer, self).__init__(**kwargs)

    def call(self, inputs: Dict[str, np.ndarray]):
        """Get LSTM encoding for action sequences"""
        x = inputs[self.feature_name]
        x = tf.reshape(x, [-1, self.action_len])
        x = self.dense_embedding(x)
        x = self.lstm(x)
        x = tf.reshape(x, [-1, self.seq_len, self.hidden_size])

        return x


class BoWActionEncodingLayer(tf.keras.layers.Layer):
    """Bag of Words encoding layer"""

    def __init__(self, feature_name: str, **kwargs):
        self.feature_name = feature_name
        super(BoWActionEncodingLayer, self).__init__()

    def call(self, inputs: Dict[str, tf.Tensor]):
        """Get bag of words features"""
        x = inputs[self.feature_name]
        return x


class WideAndDeepIRT(tf.keras.Model):
    """Wide and Deep IRT for EDM Cup 2023
    Examples:
    ---------
    >>> model = WideAndDeepIRT(hidden_size=8, dropout=0.2, unit_test_item_size=1837, seq_len=200)
    >>> deep_features = [
    ...     'history_recency',
    ...     'history_actions',
    ...     'history_item_response_time_quantile_10',
    ...     'history_item_success_rates',
    ...     'history_item_bert_embeddings',
    ...     'history_missing']
    >>> for feature in  deep_features:
    ...     model.register_feature(feature) # this allows the model to be aware of input features
    >>> categorical_feature_embeddings = {
    ...     'history_actions': tf.keras.layers.Embedding(10, 1),
    ...     'history_recency': tf.keras.layers.Embedding(200, 1),
    ...     'history_item_response_time_quantile_10': tf.keras.layers.Embedding(11, 1),
    ...     'history_missing': tf.keras.layers.Embedding(2, 8),
    ...     }
    >>> for feature, embedding_layer in categorical_feature_embeddings.items():
    ...     model.register_categorical_feature_embedding_layer(feature, embedding_layer)
    >>> model.compile(optimizer='adam', loss='binary_crossentropy')
    >>> model.fit(train_data, epochs=5, validation_data=val_data) # train_data and val_data is a tf.data.Dataset
    """

    def __init__(
        self,
        hidden_size,
        dropout,
        unit_test_item_size,
        **kwargs,
    ):
        super(WideAndDeepIRT, self).__init__(**kwargs)

        # customer registered categorical features
        self.categorical_feature_embeddings_layers = {}

        # features
        self.deep_features = []

        # layer to model the difficulty of item
        initializer = tf.keras.initializers.Zeros()

        self.item_beta_weights = self.add_weight(
            shape=[unit_test_item_size],
            initializer=initializer,
            name="item_beta_weights",
        )

        self.item_guessing_weights = self.add_weight(
            shape=[unit_test_item_size],
            initializer=initializer,
            name="item_guessing_weights",
        )

        self.dense_layer1 = tf.keras.layers.EinsumDense(
            "...x,xy->...y",
            output_shape=hidden_size,
            activation="relu",
            name="dense_layer1",
        )

        self.dense_layer2 = tf.keras.layers.EinsumDense(
            "...x,xy->...y",
            output_shape=hidden_size // 2,
            activation="relu",
            name="dense_layer2",
        )

        self.dense_layer3 = tf.keras.layers.EinsumDense(
            "...x,xy->...y", output_shape=1, activation="linear", name="dense_layer3"
        )

        self.params = {
            "hidden_size": hidden_size,
            "dropout": dropout,
            "unit_test_item_size": unit_test_item_size,
        }

    def wide_component(
        self,
        stu_ability: tf.Tensor,
        item_difficulty: tf.Tensor,
        item_guess_rate: tf.Tensor,
        is_multiple_choice: tf.Tensor,
    ):
        """Compute wide component"""
        x = tf.nn.sigmoid(stu_ability - item_difficulty)
        y = (1 - item_guess_rate) * x + item_guess_rate
        z = is_multiple_choice * y + (1 - is_multiple_choice) * x
        return z

    def deep_component(
        self, features: tf.Tensor, mask: tf.Tensor, training: bool = True
    ):
        """Compute deep component"""
        padding_locations = tf.cast(mask, dtype=tf.float32)
        padding_bias = padding_locations * _NEG_INF_FP32
        weights = tf.nn.softmax(padding_bias, axis=-1)

        x = self.dense_layer1(features)
        if training:
            x = tf.nn.dropout(x, rate=self.params["dropout"])

        x = self.dense_layer2(x)
        if training:
            x = tf.nn.dropout(x, rate=self.params["dropout"])
        x = self.dense_layer3(x)

        x = tf.reduce_sum(x, axis=-1, keepdims=False)  # remove the last dimension
        x *= weights  # remove padding, and take mean

        return tf.reduce_sum(x, axis=-1, keepdims=True)

    def call(
        self, inputs: Dict, training: bool = True, mask: Optional[tf.Tensor] = None
    ):
        """Get students' predicted probability of answering unit test questions right
        Parameters
        ----------
        inputs: Dict[str, tf.Tensor]
            input dictionary
        training: bool
            if True, activate dropout layers
        mask: tf.Tensor
            mask for the inputs, not used here. Just to comply to tf.keras.Model.call signature
        Returns
        -------
        predicted_probability: tf.Tensor
            predicted probability of correctly answering unit test questions right,
            shape = [batch_size, 1], dtype = tf.float32
        """

        # gather deep features
        deep_features = self.gather_feature_values(inputs)

        student_ability = self.deep_component(
            deep_features,
            inputs["mask"],
            training=training,
        )

        return self.wide_component(
            stu_ability=student_ability,
            item_difficulty=self.get_item_difficulty(items=inputs["future_item"]),
            is_multiple_choice=inputs["future_item_is_multiple_choice"],
            item_guess_rate=self.get_item_guessing_weights(inputs["future_item"]),
        )

    def register_categorical_feature_embedding_layer(
        self, feature_name: str, embedding_layer: tf.keras.layers.Layer
    ):
        """Associate a feature name with an embedding layer"""
        assert (
            feature_name not in self.categorical_feature_embeddings_layers
        ), f"There is already an embedding layer registered for {feature_name}"
        assert isinstance(embedding_layer, tf.keras.layers.Layer)

        self.categorical_feature_embeddings_layers[feature_name] = embedding_layer

    def register_feature(self, feature_name: str):
        """Register the feature with the model, This has been done before training"""
        assert (
            feature_name not in self.deep_features
        ), f"{feature_name} has already been registered"
        self.deep_features.append(feature_name)

    def gather_feature_values(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Gather feature values"""

        feature_vals = []

        for feature_name in self.deep_features:
            if feature_name in self.categorical_feature_embeddings_layers:
                feature_vals.append(
                    self.categorical_feature_embeddings_layers[feature_name](
                        inputs[feature_name]
                    ),
                )
            else:
                feature_vals.append(inputs[feature_name])

        # shape = [batch_size, seq_len, total_feature_dimension_size]
        return tf.concat(feature_vals, axis=-1)

    def get_item_difficulty(self, items: tf.Tensor) -> tf.Tensor:
        """Get item difficulties
        Parameters
        ----------
        items: tf.Tensor
            item index. shape = [batch_size, seq_len], dtype = tf.int32/64
        Returns
        -------
        items_difficulty: tf.Tensor
            item difficulty, shape = [batch_size, seq_len], dtype = tf.float32
        """
        item_params = self.item_beta_weights
        item_idx_params = tf.gather(item_params, items)

        return item_idx_params

    def get_item_guessing_weights(self, items: tf.Tensor) -> tf.Tensor:
        """Get item guessing weights
        Parameters
        ----------
        items: tf.Tensor
            item index. shape = [batch_size, seq_len], dtype = tf.int32/64
        Returns
        -------
        items_guessing_weights: tf.Tensor
            item guessing weights, shape = [batch_size, seq_len], dtype = tf.float32
        """

        item_params = self.item_guessing_weights
        item_idx_params = tf.gather(item_params, items)

        # make sure the guess rate is in the range of (0, 1)
        return tf.nn.sigmoid(item_idx_params)
