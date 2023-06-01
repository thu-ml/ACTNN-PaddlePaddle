import paddle
from actnn.conf import config
from actnn.qscheme import QScheme
import actnn.cpp_extension.minimax as ext_minimax
import actnn.cpp_extension.calc_precision as ext_calc_precision

class QBNScheme(QScheme):
    layers = []

    def __init__(self, group=0):
        self.initial_bits = config.initial_bits
        self.bits = config.activation_compression_bits[group]
        QBNScheme.layers.append(self)
        if len(QScheme.layers) > 0:
            self.prev_linear = QScheme.layers[-1]
        else:
            self.prev_linear = None

    def compute_quantization_bits(self, input):
        self.prev_linear = QScheme.prev_layer
        N = input.shape[0]
        D = input.shape[1]
        input_flatten = input.reshape(N, -1)
        num_features = input_flatten.shape[1]
        num_pixels = num_features // D
        # Compute min, max by groups
        if num_features % config.group_size != 0:
            # Padding
            new_num_features = (num_features // config.group_size + 1) * config.group_size
            delta = new_num_features - num_features
            input_flatten = paddle.concat([input_flatten,
                                       paddle.zeros([N, delta], dtype=input.dtype, device=input.device)], 1)
        input_groups = input_flatten.reshape([-1, config.group_size])
        mn, mx = ext_minimax.minimax(input_groups)
        if not config.pergroup:    # No per group quantization
            mn = paddle.ones_like(mn) * mn.min()
            mx = paddle.ones_like(mx) * mx.max()
        # Average range over pixels [N]
        Range_sqr = paddle.norm((mx - mn).reshape(N, -1), axis=1).astype('float32').square() * (config.group_size / num_pixels)
        # greedy
        C = Range_sqr.astype('float32').cpu()
        b = paddle.ones(N, dtype='int32') * self.initial_bits
        w = paddle.ones(N, dtype='int32')
        b = ext_calc_precision.calc_precision(b, C, w, int(self.bits * N))
        return input_groups.reshape(N, -1, config.group_size), b.cuda(), mn.reshape(N, -1, 1), mx.reshape(N, -1, 1)

    @staticmethod
    def allocate_perlayer():
        for layer in QBNScheme.layers:
            if layer.prev_linear is not None:
                layer.bits = layer.prev_linear.bits

            else:
                layer.bits = 8

