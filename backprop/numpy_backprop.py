"""Manual NumPy forward/backward/SGD for probes of pooled SAE features.

Samples occupy the last axis, as in the user's class code. A classifier is a
probe, not a generator intervention. No PyTorch/autograd is used here.
"""
import numpy as np


def init_model(input_size, output_size=2, hidden_size=0, seed=0):
    if input_size < 1 or output_size < 2 or hidden_size < 0:
        raise ValueError('Need positive input size, >=2 classes, and nonnegative hidden size')
    rng = np.random.default_rng(seed)
    sizes = [input_size] + ([hidden_size] if hidden_size else []) + [output_size]
    layers = []
    for i, (nin, nout) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(dict(W=rng.normal(0, np.sqrt(2 / nin), (nout, nin)),
                           b=np.zeros((nout, 1)), activation='relu' if i < len(sizes) - 2 else 'linear'))
    return dict(layers=layers, input_size=input_size, output_size=output_size)


def inference(model, inputs):
    """Return logits and a cache; softmax is fused with cross-entropy below."""
    current = np.asarray(inputs, dtype=np.float64)
    if current.ndim != 2 or current.shape[0] != model['input_size']:
        raise ValueError('Expected [input_features, batch_size]')
    cache = []
    for layer in model['layers']:
        pre = layer['W'] @ current + layer['b']
        cache.append(dict(input=current, pre=pre))
        current = np.maximum(pre, 0) if layer['activation'] == 'relu' else pre
    return current, cache


def loss_crossentropy(logits, labels):
    """Stable mean cross-entropy and derivative with respect to logits."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != logits.shape[1] or not len(labels):
        raise ValueError('Expected one integer label per sample and a nonempty batch')
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0) or np.any(labels >= logits.shape[0]):
        raise ValueError('Labels must be integer class indices')
    shifted = logits - logits.max(axis=0, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=0, keepdims=True))
    loss = -logp[labels, np.arange(len(labels))].mean()
    dv = np.exp(logp)
    dv[labels, np.arange(len(labels))] -= 1
    return float(loss), dv / len(labels)


def calc_gradient(model, cache, dv_output):
    """Return parameter gradients AND input gradient for feature attribution."""
    dv = np.asarray(dv_output, dtype=np.float64)
    grads = [None] * len(model['layers'])
    for i in range(len(grads) - 1, -1, -1):
        layer, act = model['layers'][i], cache[i]
        if layer['activation'] == 'relu':
            dv = dv * (act['pre'] > 0)
        grads[i] = dict(W=dv @ act['input'].T, b=dv.sum(axis=1, keepdims=True))
        dv = layer['W'].T @ dv
    return grads, dv


def update_weights(model, grads, learning_rate=.01, weight_decay=0.):
    """SGD; L2 weight decay applies to W only, matching the class implementation."""
    for layer, grad in zip(model['layers'], grads):
        layer['W'] -= learning_rate * (grad['W'] + weight_decay * layer['W'])
        layer['b'] -= learning_rate * grad['b']
    return model


def positive_logit_gradient(model, inputs, positive_class=1):
    """Gradient of positive-vs-negative logit margin for a binary attribute."""
    logits, cache = inference(model, inputs)
    if logits.shape[0] != 2 or positive_class not in (0, 1):
        raise ValueError('Attribute attribution requires two classes')
    dv = np.zeros_like(logits)
    dv[positive_class] = 1
    dv[1 - positive_class] = -1
    return calc_gradient(model, cache, dv)[1]


def save_model(path, model, mean, std, metadata):
    import json
    arrays = dict(mean=mean, std=std, metadata=np.array(json.dumps(metadata)),
                  sizes=np.array([model['input_size']] + [l['W'].shape[0] for l in model['layers']]))
    for i, layer in enumerate(model['layers']):
        arrays[f'W{i}'], arrays[f'b{i}'] = layer['W'], layer['b']
    np.savez_compressed(path, **arrays)


def load_model(path):
    import json
    with np.load(path, allow_pickle=False) as data:
        sizes = data['sizes']
        model = init_model(int(sizes[0]), int(sizes[-1]), int(sizes[1]) if len(sizes) == 3 else 0)
        for i, layer in enumerate(model['layers']):
            layer['W'], layer['b'] = data[f'W{i}'].copy(), data[f'b{i}'].copy()
        return model, data['mean'].copy(), data['std'].copy(), json.loads(str(data['metadata']))
