"""Prepare loaded weights for the approved FP16/FP32 compiled inference."""
def prepare_model(model):
    """Fuse hidden-state RMS normalization while keeping FP32 accumulation.

    FP16 nonzero states are far above 1e-12. A tiny epsilon handles zero
    states without changing the original normalization in this range.
    The FP32 spectral band normalization retains its original clamp.
    """
    import mlx.core as mx
    import mlx.nn as nn

    model.set_dtype(mx.float16)
    model.mask_estimators[0].prepare()

    def replace(source):
        target = nn.RMSNorm(source.weight.shape[0], eps=1e-24)
        target.weight = source.weight
        return target

    for pair in model.layers:
        for transformer in pair:
            transformer.norm = replace(transformer.norm)
            for attention, feedforward in transformer.layers:
                attention.prepare()
                attention.norm = replace(attention.norm)
                feedforward.net[0] = replace(feedforward.net[0])
