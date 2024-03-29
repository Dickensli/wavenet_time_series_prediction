import numpy as np
import tensorflow as tf

from ops import causal_conv, mu_law_encode,sequence_dense_layer
from base_model import baseNN
from data_read import DataReader
import time
import os

def selu(x):
    """
    SELU activation
    https://arxiv.org/abs/1706.02515
    :param x:
    :return:
    """
    with tf.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))
    
def create_variable(name, shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name,dtype=tf.float32)
    return variable


def create_embedding_table(name, shape):
    if shape[0] == shape[1]:
        # Make a one-hot encoding as the initial value.
        initial_val = np.identity(n=shape[0], dtype=np.float32)
        return tf.Variable(initial_val, name=name)
    else:
        return create_variable(name, shape)


def create_bias_variable(name, shape):
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)


class WaveNetModel(baseNN):
    '''Implements the WaveNet network for generative audio.

    Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(N)] * M
        filter_width = 2  # Convolutions just use 2 samples.
        residual_channels = 16  # Not specified in the paper.
        dilation_channels = 32  # Not specified in the paper.
        skip_channels = 16      # Not specified in the paper.
        net = WaveNetModel(batch_size, dilations, filter_width,
                           residual_channels, dilation_channels,
                           skip_channels)
        loss = net.loss(input_batch)
    '''

    def __init__(self,
                 batch,
                 dilations,
                 filter_width,
                 residual_channels,
                 dilation_channels,
                 skip_channels,
                 quantization_channels=2**8,
                 use_biases=False,
                 scalar_input=False,
                 initial_filter_width=32,
                 initial_channel=32,
                 histograms=False,
                 global_condition_channels=None,
                 global_condition_cardinality=None,
                 num_step=288,
                 feature_dim=2,
                 label_dim=1,
                 seq_len=2048,
                 test_using_predict=True,
                 single=True,
                 ** kwargs):
        '''Initializes the WaveNet model.

        Args:
            batch_size: How many audio files are supplied per batch
                (recommended: 1).
            dilations: A list with the dilation factor for each layer.
            filter_width: The samples that are included in each convolution,
                after dilating.
            residual_channels: How many filters to learn for the residual.
            dilation_channels: How many filters to learn for the dilated
                convolution.
            skip_channels: How many filters to learn that contribute to the
                quantized softmax output.
            quantization_channels: How many amplitude values to use for audio
                quantization and the corresponding one-hot encoding.
                Default: 256 (8-bit quantization).
            use_biases: Whether to add a bias layer to each convolution.
                Default: False.
            scalar_input: Whether to use the quantized waveform directly as
                input to the network instead of one-hot encoding it.
                Default: False.
            initial_filter_width: The width of the initial filter of the
                convolution applied to the scalar input. This is only relevant
                if scalar_input=True.
            histograms: Whether to store histograms in the summary.
                Default: False.
            global_condition_channels: Number of channels in (embedding
                size) of global conditioning vector. None indicates there is
                no global conditioning.
            global_condition_cardinality: Number of mutually exclusive
                categories to be embedded in global condition embedding. If
                not None, then this implies that global_condition tensor
                specifies an integer selecting which of the N global condition
                categories, where N = global_condition_cardinality. If None,
                then the global_condition tensor is regarded as a vector which
                must have dimension global_condition_channels.

        '''
        self.batchsize = batch
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.quantization_channels = quantization_channels
        self.use_biases = use_biases
        self.skip_channels = skip_channels
        self.scalar_input = scalar_input
        self.initial_filter_width = initial_filter_width
        self.initial_channel = initial_channel
        self.histograms = histograms
        self.global_condition_channels = global_condition_channels
        self.global_condition_cardinality = global_condition_cardinality
        self.feature_dim = feature_dim
        self.label_dim = label_dim
        self.seq_len = seq_len
        self.single = single
        self.num_step = num_step
        self.test_using_predict = test_using_predict
        self.input_seq()
        self.receptive_field = WaveNetModel.calculate_receptive_field(
            self.filter_width, self.dilations, self.scalar_input,
            self.initial_filter_width)
        self.variables = self._create_variables()
        super(WaveNetModel, self).__init__(**kwargs)

    @staticmethod
    def calculate_receptive_field(filter_width, dilations, scalar_input,
                                  initial_filter_width):
        """
        calculate receptive field
        :param filter_width: filter width
        :param dilations: dilation to expand the area
        :param scalar_input: bool
        :param initial_filter_width: initial filter width
        :return: receptive fild
        """
        receptive_field = (filter_width - 1) * sum(dilations) + 1
        if scalar_input:
            receptive_field += initial_filter_width - 1
        else:
            receptive_field += filter_width - 1
        return receptive_field

    def transform(self, data, mean):
        """
        prdict the diff value
        :param data: true value
        :param mean: mean value
        :return: diff value
        """
        return data - mean

    def inverse_mean_transform(self, x, mean):
        """
        length of x is different with mean, so we need to slice  expm1
        :param x:
        :param mean:
        :return: true value
        """
        length = tf.shape(mean)[1] - tf.shape(x)[1]
        slice_mean = tf.slice(mean, [0, length, 0], [-1, -1, -1])
        return tf.exp(x + slice_mean) - 1

    def inverse_transform(self, x):
        return tf.exp(x) - 1

    def input_seq(self):
        """
        init the placeholder data,d7mean,encode
        :return: none
        """
        self.keep_prob = tf.placeholder(tf.float32)
        self.label = tf.placeholder(tf.float32, [None, None, self.label_dim])
        if self.single:
            self.data = tf.placeholder(tf.float32, [None, None, self.feature_dim])
            self.d7mean = tf.placeholder(tf.float32, [None, None, self.feature_dim])
            self.encode = tf.concat([self.transform(self.data, self.d7mean),
                                     self.d7mean],
                                    axis=-1)
        else:
            self.data = tf.placeholder(tf.float32, [None, None, self.feature_dim])
            self.d1max = tf.placeholder(tf.float32, [None, None, self.feature_dim])
            self.d7max = tf.placeholder(tf.float32, [None, None, self.feature_dim])
            self.d7mean = tf.placeholder(tf.float32, [None, None, self.feature_dim])
            self.d4mean = tf.placeholder(tf.float32, [None, None, self.feature_dim])
            self.minute = tf.placeholder(tf.float32, [None, None, self.feature_dim])
            self.week = tf.placeholder(tf.float32, [None, None, self.feature_dim])
            self.encode = tf.concat([self.transform(self.data, self.d7mean),
                                     self.d7mean,
                                     self.d1max,
                                     self.d7max,
                                     self.d4mean,
                                     self.minute,
                                     self.week],
                                    axis=-1)

    def _create_variables(self):
        '''This function creates all variables used by the network.
        This allows us to share them between multiple calls to the loss
        function and generation function.'''

        var = dict()

        with tf.variable_scope('wavenet'):

            if self.global_condition_cardinality is not None:
                # We only look up the embedding if we are conditioning on a
                # set of mutually-exclusive categories. We can also condition
                # on an already-embedded dense vector, in which case it's
                # given to us and we don't need to do the embedding lookup.
                # Still another alternative is no global condition at all, in
                # which case we also don't do a tf.nn.embedding_lookup.
                with tf.variable_scope('embeddings'):
                    layer = dict()
                    layer['gc_embedding'] = create_embedding_table(
                        'gc_embedding',
                        [self.global_condition_cardinality,
                         self.global_condition_channels])
                    var['embeddings'] = layer

            with tf.variable_scope('causal_layer'):
                layer = dict()
                if self.scalar_input:
                    initial_channels = 1
                    initial_filter_width = self.initial_filter_width
                else:
                    initial_filter_width = self.filter_width
                layer['filter'] = create_variable(
                    'filter',
                    [initial_filter_width,
                     self.initial_channel,
                     self.residual_channels])
                var['causal_layer'] = layer

            var['dilated_stack'] = list()
            with tf.variable_scope('dilated_stack'):
                for i, dilation in enumerate(self.dilations):
                    with tf.variable_scope('layer{}'.format(i)):
                        current = dict()
                        current['filter'] = create_variable(
                            'filter',
                            [self.filter_width,
                             self.residual_channels,
                             self.dilation_channels])
                        current['gate'] = create_variable(
                            'gate',
                            [self.filter_width,
                             self.residual_channels,
                             self.dilation_channels])
                        current['dense'] = create_variable(
                            'dense',
                            [1,
                             self.dilation_channels,
                             self.residual_channels])
                        current['skip'] = create_variable(
                            'skip',
                            [1,
                             self.dilation_channels,
                             self.skip_channels])

                        if self.global_condition_channels is not None:
                            current['gc_gateweights'] = create_variable(
                                'gc_gate',
                                [1, self.global_condition_channels,
                                 self.dilation_channels])
                            current['gc_filtweights'] = create_variable(
                                'gc_filter',
                                [1, self.global_condition_channels,
                                 self.dilation_channels])

                        if self.use_biases:
                            current['filter_bias'] = create_bias_variable(
                                'filter_bias',
                                [self.dilation_channels])
                            current['gate_bias'] = create_bias_variable(
                                'gate_bias',
                                [self.dilation_channels])
                            current['dense_bias'] = create_bias_variable(
                                'dense_bias',
                                [self.residual_channels])
                            current['skip_bias'] = create_bias_variable(
                                'slip_bias',
                                [self.skip_channels])

                        var['dilated_stack'].append(current)

            with tf.variable_scope('postprocessing'):
                current = dict()
                current['postprocess1'] = create_variable(
                    'postprocess1',
                    [1, self.skip_channels, self.skip_channels])
                current['postprocess2'] = create_variable(
                    'postprocess2',
                    [1, self.skip_channels, self.quantization_channels])
                if self.use_biases:
                    current['postprocess1_bias'] = create_bias_variable(
                        'postprocess1_bias',
                        [self.skip_channels])
                    current['postprocess2_bias'] = create_bias_variable(
                        'postprocess2_bias',
                        [self.quantization_channels])
                var['postprocessing'] = current

        return var

    def _create_causal_layer(self, input_batch):
        '''Creates a single causal convolution layer.

        The layer can change the number of channels.
        '''
        with tf.name_scope('causal_layer'):
            weights_filter = self.variables['causal_layer']['filter']
            #print weights_filter.shape
            return causal_conv(input_batch, weights_filter, 1)

    def _create_dilation_layer(self, input_batch, layer_index, dilation,
                               global_condition_batch, output_width):
        '''Creates a single causal dilated convolution layer.

        Args:
             input_batch: Input to the dilation layer.
             layer_index: Integer indicating which layer this is.
             dilation: Integer specifying the dilation size.
             global_conditioning_batch: Tensor containing the global data upon
                 which the output is to be conditioned upon. Shape:
                 [batch size, 1, channels]. The 1 is for the axis
                 corresponding to time so that the result is broadcast to
                 all time steps.

        The layer contains a gated filter that connects to dense output
        and to a skip connection:

               |-> [gate]   -|        |-> 1x1 conv -> skip output
               |             |-> (*) -|
        input -|-> [filter] -|        |-> 1x1 conv -|
               |                                    |-> (+) -> dense output
               |------------------------------------|

        Where `[gate]` and `[filter]` are causal convolutions with a
        non-linear activation at the output. Biases and global conditioning
        are omitted due to the limits of ASCII art.

        '''
        variables = self.variables['dilated_stack'][layer_index]

        weights_filter = variables['filter']
        weights_gate = variables['gate']

        #weights_filter = tf.Print(weights_filter,["weights_filter:",weights_filter])
        conv_filter = causal_conv(input_batch, weights_filter, dilation)
        conv_gate = causal_conv(input_batch, weights_gate, dilation)

        if global_condition_batch is not None:
            weights_gc_filter = variables['gc_filtweights']
            conv_filter = conv_filter + tf.nn.conv1d(global_condition_batch,
                                                     weights_gc_filter,
                                                     stride=1,
                                                     padding="SAME",
                                                     name="gc_filter")
            weights_gc_gate = variables['gc_gateweights']
            conv_gate = conv_gate + tf.nn.conv1d(global_condition_batch,
                                                 weights_gc_gate,
                                                 stride=1,
                                                 padding="SAME",
                                                 name="gc_gate")

        if self.use_biases:
            filter_bias = variables['filter_bias']
            gate_bias = variables['gate_bias']
            conv_filter = tf.add(conv_filter, filter_bias)
            conv_gate = tf.add(conv_gate, gate_bias)

        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

        # The 1x1 conv to produce the residual output
        weights_dense = variables['dense']
        transformed = tf.nn.conv1d(
            out, weights_dense, stride=1, padding="SAME", name="dense")

        # The 1x1 conv to produce the skip output
        skip_cut = tf.shape(out)[1] - output_width
        out_skip = tf.slice(out, [0, skip_cut, 0], [-1, -1, -1])
        out_skip = out_skip
        weights_skip = variables['skip']
        skip_contribution = tf.nn.conv1d(
            out_skip, weights_skip, stride=1, padding="SAME", name="skip")

        if self.use_biases:
            dense_bias = variables['dense_bias']
            skip_bias = variables['skip_bias']
            transformed = transformed + dense_bias
            skip_contribution = skip_contribution + skip_bias

        if self.histograms:
            layer = 'layer{}'.format(layer_index)
            tf.histogram_summary(layer + '_filter', weights_filter)
            tf.histogram_summary(layer + '_gate', weights_gate)
            tf.histogram_summary(layer + '_dense', weights_dense)
            tf.histogram_summary(layer + '_skip', weights_skip)
            if self.use_biases:
                tf.histogram_summary(layer + '_biases_filter', filter_bias)
                tf.histogram_summary(layer + '_biases_gate', gate_bias)
                tf.histogram_summary(layer + '_biases_dense', dense_bias)
                tf.histogram_summary(layer + '_biases_skip', skip_bias)

        input_cut = tf.shape(input_batch)[1] - tf.shape(transformed)[1]
        input_batch = tf.slice(input_batch, [0, input_cut, 0], [-1, -1, -1])
        self.denseout = input_batch+transformed
        self.skip = skip_contribution
        return skip_contribution, input_batch + transformed

    def _generator_conv(self, input_batch, state_batch, weights):
        '''Perform convolution for a single convolutional processing step.'''
        # TODO generalize to filter_width > 2
        past_weights = weights[0, :, :]
        curr_weights = weights[1, :, :]
        output = tf.matmul(state_batch, past_weights) + tf.matmul(
            input_batch, curr_weights)
        return output

    def _generator_causal_layer(self, input_batch, state_batch):
        with tf.name_scope('causal_layer'):
            weights_filter = self.variables['causal_layer']['filter']
            output = self._generator_conv(
                input_batch, state_batch, weights_filter)
        return output

    def _generator_dilation_layer(self, input_batch, state_batch, layer_index,
                                  dilation, global_condition_batch):
        variables = self.variables['dilated_stack'][layer_index]

        weights_filter = variables['filter']
        weights_gate = variables['gate']
        output_filter = self._generator_conv(
            input_batch, state_batch, weights_filter)
        output_gate = self._generator_conv(
            input_batch, state_batch, weights_gate)

        if global_condition_batch is not None:
            global_condition_batch = tf.reshape(global_condition_batch,
                                                shape=(1, -1))
            weights_gc_filter = variables['gc_filtweights']
            weights_gc_filter = weights_gc_filter[0, :, :]
            output_filter += tf.matmul(global_condition_batch,
                                       weights_gc_filter)
            weights_gc_gate = variables['gc_gateweights']
            weights_gc_gate = weights_gc_gate[0, :, :]
            output_gate += tf.matmul(global_condition_batch,
                                     weights_gc_gate)

        if self.use_biases:
            output_filter = output_filter + variables['filter_bias']
            output_gate = output_gate + variables['gate_bias']

        out = tf.tanh(output_filter) * tf.sigmoid(output_gate)

        weights_dense = variables['dense']
        transformed = tf.matmul(out, weights_dense[0, :, :])
        if self.use_biases:
            transformed = transformed + variables['dense_bias']

        weights_skip = variables['skip']
        skip_contribution = tf.matmul(out, weights_skip[0, :, :])
        if self.use_biases:
            skip_contribution = skip_contribution + variables['skip_bias']

        return skip_contribution, input_batch + transformed

    def _create_network(self, input_batch, global_condition_batch):
        '''Construct the WaveNet network.'''
        outputs = []
        current_layer = input_batch

        # Pre-process the input with a regular convolution
        with tf.variable_scope('wavenet'):
            # feature dim  to inital channel
            current_layer = sequence_dense_layer(current_layer, self.initial_channel, bias=True, activation=tf.nn.tanh,
                                         batch_norm=None,
                                         dropout=self.keep_prob, scope='input_dense', reuse=tf.AUTO_REUSE)
        #current_layer=tf.Print(current_layer,["1:",tf.shape(current_layer)])
        current_layer = self._create_causal_layer(current_layer)
        #current_layer = tf.Print(current_layer, ["2:", tf.shape(current_layer)])
        self.casual = current_layer

        output_width = tf.shape(input_batch)[1] - self.receptive_field + 1
        self.width = output_width
        # Add all defined dilation layers.
        self.shape_stack = np.array([])
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    output, current_layer = self._create_dilation_layer(
                        current_layer, layer_index, dilation,
                        global_condition_batch, output_width)
                    outputs.append(output)
        self.outputs = np.array(outputs)
        with tf.name_scope('postprocessing'):
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            w1 = self.variables['postprocessing']['postprocess1']
            w2 = self.variables['postprocessing']['postprocess2']
            if self.use_biases:
                b1 = self.variables['postprocessing']['postprocess1_bias']
                b2 = self.variables['postprocessing']['postprocess2_bias']

            if self.histograms:
                tf.histogram_summary('postprocess1_weights', w1)
                tf.histogram_summary('postprocess2_weights', w2)
                if self.use_biases:
                    tf.histogram_summary('postprocess1_biases', b1)
                    tf.histogram_summary('postprocess2_biases', b2)

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            transformed1 = tf.nn.relu(total)
            self.transformed1 = transformed1
            conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
            if self.use_biases:
                conv1 = tf.add(conv1, b1)
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")
            if self.use_biases:
                conv2 = tf.add(conv2, b2)
        #conv2 =tf.Print(conv2,[tf.shape(conv2),'any thing i want'],message='Debug message:',summarize=100)

        return conv2

    def _create_generator(self, input_batch, global_condition_batch):
        '''Construct an efficient incremental generator.'''
        init_ops = []
        push_ops = []
        outputs = []
        current_layer = input_batch
        with tf.variable_scope('wavenet'):
            current_layer = sequence_dense_layer(current_layer, self.initial_channel, bias=True, activation=tf.nn.tanh,
                                             batch_norm=None,
                                             dropout=self.keep_prob, scope='input_dense', reuse=True)
        current_layer = tf.reshape(current_layer,[tf.shape(current_layer)[0],self.initial_channel])
        q = tf.FIFOQueue(
            1,
            dtypes=tf.float32,
            shapes=(self.batchsize, self.initial_channel))
        init = q.enqueue_many(
            tf.zeros((1, self.batchsize, self.initial_channel)))

        current_state = q.dequeue()
        push = q.enqueue([current_layer])
        init_ops.append(init)
        push_ops.append(push)

        current_layer = self._generator_causal_layer(
                            current_layer, current_state)

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):

                    q = tf.FIFOQueue(
                        dilation,
                        dtypes=tf.float32,
                        shapes=(self.batchsize, self.residual_channels))
                    init = q.enqueue_many(
                        tf.zeros((dilation, self.batchsize,
                                  self.residual_channels)))

                    current_state = q.dequeue()
                    push = q.enqueue([current_layer])
                    init_ops.append(init)
                    push_ops.append(push)

                    output, current_layer = self._generator_dilation_layer(
                        current_layer, current_state, layer_index, dilation,
                        global_condition_batch)
                    outputs.append(output)
        self.init_ops = init_ops
        self.push_ops = push_ops

        with tf.name_scope('postprocessing'):
            variables = self.variables['postprocessing']
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            w1 = variables['postprocess1']
            w2 = variables['postprocess2']
            if self.use_biases:
                b1 = variables['postprocess1_bias']
                b2 = variables['postprocess2_bias']

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            transformed1 = tf.nn.relu(total)
            #transformed1=tf.Print(transformed1,["transform1:" ,tf.shape(transformed1)])
            conv1 = tf.matmul(transformed1, w1[0, :, :])
            if self.use_biases:
                conv1 = conv1 + b1
            transformed2 = tf.nn.relu(conv1)
            #transformed2=tf.Print(transformed2, ["transform2:", tf.shape(transformed2)])
            conv2 = tf.matmul(transformed2, w2[0, :, :])
            if self.use_biases:
                conv2 = conv2 + b2

        return conv2

    def _one_hot(self, input_batch):
        '''One-hot encodes the waveform amplitudes.

        This allows the definition of the network as a categorical distribution
        over a finite set of possible amplitudes.
        '''
        with tf.name_scope('one_hot_encode'):
            encoded = tf.one_hot(
                input_batch,
                depth=self.quantization_channels,
                dtype=tf.float32)
            shape = [self.batchsize, -1, self.quantization_channels]
            encoded = tf.reshape(encoded, shape)
        return encoded

    def _embed_gc(self, global_condition):
        '''Returns embedding for global condition.
        :param global_condition: Either ID of global condition for
               tf.nn.embedding_lookup or actual embedding. The latter is
               experimental.
        :return: Embedding or None
        '''
        embedding = None
        if self.global_condition_cardinality is not None:
            # Only lookup the embedding if the global condition is presented
            # as an integer of mutually-exclusive categories ...
            embedding_table = self.variables['embeddings']['gc_embedding']
            embedding = tf.nn.embedding_lookup(embedding_table,
                                               global_condition)
        elif global_condition is not None:
            # ... else the global_condition (if any) is already provided
            # as an embedding.

            # In this case, the number of global_embedding channels must be
            # equal to the the last dimension of the global_condition tensor.
            gc_batch_rank = len(global_condition.get_shape())
            dims_match = (global_condition.get_shape()[gc_batch_rank - 1] ==
                          self.global_condition_channels)
            if not dims_match:
                raise ValueError('Shape of global_condition {} does not'
                                 ' match global_condition_channels {}.'.
                                 format(global_condition.get_shape(),
                                        self.global_condition_channels))
            embedding = global_condition

        if embedding is not None:
            embedding = tf.reshape(
                embedding,
                [self.batchsize, 1, self.global_condition_channels])

        return embedding

    def predict_proba(self, waveform, global_condition=None, name='wavenet'):
        '''Computes the probability distribution of the next sample based on
        all samples in the input waveform.
        If you want to generate audio by feeding the output of the network back
        as an input, see predict_proba_incremental for a faster alternative.'''
        with tf.name_scope(name):
            if self.scalar_input:
                encoded = tf.cast(waveform, tf.float32)
                encoded = tf.reshape(encoded, [-1, 1])
            else:
                encoded = self._one_hot(waveform)

            gc_embedding = self._embed_gc(global_condition)
            raw_output = self._create_network(encoded, gc_embedding)
            self.raw_input = encoded
            self.raw_output = raw_output
            out = tf.reshape(raw_output, [-1, self.quantization_channels])

            # Cast to float64 to avoid bug in TensorFlow
            proba = tf.cast(
                tf.nn.softmax(tf.cast(out, tf.float64)), tf.float32)
            self.last= proba
            last = tf.slice(
                proba,
                [tf.shape(proba)[0] - 1, 0],
                [1, self.quantization_channels])
            return tf.reshape(last, [-1])

    def predict_proba_incremental(self, waveform, global_condition=None,
                                  name='wavenet'):
        '''Computes the probability distribution of the next sample
        incrementally, based on a single sample and all previously passed
        samples.'''
        if self.filter_width > 2:
            raise NotImplementedError("Incremental generation does not "
                                      "support filter_width > 2.")
        if self.scalar_input:
            raise NotImplementedError("Incremental generation does not "
                                      "support scalar input yet.")
        with tf.name_scope(name):
            raw_output = self._create_generator(waveform, None)
            return raw_output

    def predict_incremental(self, waveform, global_condition=None,
                                  name='wavenet'):
        '''Computes the probability distribution of the next sample
        incrementally, based on a single sample and all previously passed
        samples.'''
        if self.filter_width > 2:
            raise NotImplementedError("Incremental generation does not "
                                      "support filter_width > 2.")
        if self.scalar_input:
            raise NotImplementedError("Incremental generation does not "
                                      "support scalar input yet.")
        with tf.name_scope(name):
            raw_output = self._create_generator(waveform, None)
            return raw_output

    def calculate_loss(self):
        '''Creates a WaveNet network and returns the  loss.

        The variables are all scoped to the given name.
        '''
        with tf.name_scope('wavenet'):
            # get the encode
            network_input = self.encode
            raw_output = self._create_network(network_input, None)
            self.raw_input = network_input
            self.raw_output = raw_output
            with tf.name_scope('loss'):

                loss = (self.sequence_smape(
                    self.inverse_transform(self.label),
                    self.inverse_mean_transform(raw_output, self.d7mean)))
                loss = tf.Print(loss, ['loss', tf.shape(loss)])
                loss_mae = tf.reduce_mean(tf.metrics.mean_absolute_error(
                    self.inverse_transform(self.label),
                    self.inverse_mean_transform(raw_output,self.d7mean)))
                reduced_loss = tf.reduce_mean(loss)
                #reduced_loss = loss
                if self.test_using_predict:
                    raw_out_using_predict = self.predict_accumulate()
                    loss_mae = tf.reduce_mean(tf.keras.losses.mean_absolute_error(
                        self.inverse_transform(self.label),
                        self.inverse_mean_transform(raw_out_using_predict, self.d7mean)))
                tf.summary.scalar('loss', reduced_loss)
                return reduced_loss,loss_mae
    def tf_mse(self,pred,true):
        return tf.square(pred-true)

    def sequence_smape(self,y, y_hat):
        y = tf.cast(y, tf.float32)
        smape = 2 * (tf.abs(y_hat - y) / (tf.abs(y) + tf.abs(y_hat)))
        return smape

    def concat_next_predict(self, encode, decode, i):
        if tf.shape(decode)[-1] == 1:
            return decode
        one_encode = tf.slice(encode, [0, i, 0], [-1, 1, -1])
        one_encode_feature = tf.slice(one_encode, [0, 0, 1], [-1, -1, -1])
        # one_encode_feature = tf.Print(one_encode_feature,["one_encode:",tf.shape(one_encode_feature)])
        next_predict = tf.concat([decode, one_encode_feature], axis=-1)
        return next_predict

    def predict_accumulate(self):
        output = []
        encode = self.encode
        cur_encode = tf.slice(encode, [0,self.seq_len, 0], [-1, -1, -1])
        # input 2048 length seq
        for i in range(0,self.seq_len-1):
            single_point = tf.slice(cur_encode, [0, i, 0], [-1, 1, -1])
            self.predict_incremental(single_point)
        for i in range(0, self.num_step):
            if i == 0:
                single_point = tf.slice(encode, [0, self.seq_len-1, 0], [-1, 1, -1])
                decode = self.predict_incremental(single_point)
                output.append(decode)
                cur_encode = self.concat_next_predict(cur_encode, decode, i)
            else:
                decode = self.predict_incremental(cur_encode)
                cur_encode = self.concat_next_predict(cur_encode, decode, i)
                output.append(decode)
        predict_out = tf.concat(output, 1)
        return predict_out

if __name__=="__main__":
    base_dir = "/Users/didi/PycharmProjects/wavenet_clound/data"
    test_width = 288
    train_width = 2048
    dilations = [2 ** i for i in range(0, 11)]
    start = time.time()
    dr = DataReader(data_dir=base_dir, file_name='test_wave.h5', test_width=test_width, train_width=train_width,
                    num=None, single=True)
    second = time.time()
    print("datareader time:", second - start)

    nn = WaveNetModel(
        batch=1,
        dilations=dilations,
        filter_width=2,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=32,
        quantization_channels=1,
        use_biases=True,
        scalar_input=False,
        initial_filter_width=32,
        initial_channel=32,
        histograms=False,
        global_condition_channels=None,
        global_condition_cardinality=None,
        num_step=test_width,
        feature_dim=1,
        label_dim=1,
        seq_len=train_width,
        test_using_predict=False,
        single=True,
        # super
        reader=dr,
        log_dir=os.path.join(base_dir, 'logs'),
        checkpoint_dir=os.path.join(base_dir, 'checkpoints'),
        predict_dir=os.path.join(base_dir, 'predictions'),
        batch_size=32,
        num_training_steps=10000,
        learning_rate=1e-5,
        optimizer='adam',
        grad_clip=1,
        clip=True,
        keep_prob=0.5,
        regularization_constant=0,
        early_stopping_steps=300,
        warm_start_init_step=0,
        num_restarts=None,
        enable_parameter_averaging=False,
        min_steps_to_checkpoint=1000,
        log_interval=1,
        loss_averaging_window=100,
        num_validation_batches=100,

    )
    third = time.time()
    print("build the model time:", third - second)
    nn.train()
    nn.restore()
    nn.predict_incre()

