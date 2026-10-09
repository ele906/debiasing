"""Finite-difference verification only. Does not train a model or use real data."""
import numpy as np
from numpy_backprop import init_model, inference, loss_crossentropy, calc_gradient


def main():
    rng = np.random.default_rng(42)
    eps = 1e-6
    for hidden in (0, 4):
        model = init_model(3, 2, hidden, seed=5)
        x = rng.normal(size=(3, 5))
        y = np.array([0, 1, 1, 0, 1])
        logits, cache = inference(model, x)
        _, dv = loss_crossentropy(logits, y)
        grads, dx = calc_gradient(model, cache, dv)

        def objective():
            return loss_crossentropy(inference(model, x)[0], y)[0]

        errors = []
        arrays = [(layer[key], grad[key]) for layer, grad in zip(model['layers'], grads) for key in ('W', 'b')]
        arrays.append((x, dx))
        for array, analytic in arrays:
            for idx in np.ndindex(array.shape):
                original = array[idx]
                array[idx] = original + eps; plus = objective()
                array[idx] = original - eps; minus = objective()
                array[idx] = original
                numeric = (plus - minus) / (2 * eps)
                errors.append(abs(numeric - analytic[idx]))
                np.testing.assert_allclose(numeric, analytic[idx], atol=1e-7, rtol=1e-5)
        print(f'hidden={hidden}: parameter/input gradients passed; max absolute error={max(errors):.3g}')
    extreme = np.array([[10000., -10000.], [-10000., 10000.]])
    loss, grad = loss_crossentropy(extreme, np.array([0, 1]))
    assert np.isfinite(loss) and np.isfinite(grad).all()
    print('Extreme-logit numerical stability passed. No training performed.')


if __name__ == '__main__':
    main()
