"""Build named model instances from a class and per-axis value grids."""

import itertools


def cartesian(clazz, **axes):
    """Instantiate every combination of the given axes as a ``{name: model}`` dict.

    Each axis is either a ``{token: value}`` dict -- naming the value where it is
    declared -- or a plain list/scalar, shorthand for ``{str(v): v}`` (clean for
    strings, e.g. ``kernel=["cubic", "gaussian"]``). The model name concatenates the
    chosen per-axis tokens, so the name is authored at the call site rather than
    scraped from each value's ``repr``.

    Args:
        clazz: The model class to instantiate.
        **axes: Per-parameter value grids (dict, list/tuple, or a single scalar).

    Returns:
        An ordered ``{name: instance}`` dict over the cartesian product of the axes.

    Raises:
        ValueError: If an axis has duplicate tokens, or two combinations resolve to the
            same model name (which would otherwise silently overwrite each other).
    """

    def _named(values):
        if isinstance(values, dict):
            return values
        items = list(values) if isinstance(values, (list, tuple)) else [values]
        tokens = [str(x) for x in items]
        if len(set(tokens)) != len(tokens):
            raise ValueError(f"cartesian got duplicate tokens in an axis: {tokens}")
        return dict(zip(tokens, items))

    axes = {k: _named(v) for k, v in axes.items()}
    out = {}
    for combo in itertools.product(*(d.items() for d in axes.values())):
        token = ",".join(t for t, _ in combo)
        config = {k: val for k, (_, val) in zip(axes, combo)}
        name = f"{clazz.__name__}[{token}]" if token else clazz.__name__
        if name in out:
            raise ValueError(f"cartesian produced a duplicate model name {name!r}; use distinct axis tokens")
        out[name] = clazz(**config)
    return out


def models_from_clazzes(*clazzes, **defaults):
    """Build one default-configured model per class (kept for downstream imports).

    Deprecated thin shim over direct construction -- prefer :func:`cartesian` to
    declare an explicit, named sweep.

    Args:
        *clazzes: Model classes to instantiate with the shared defaults.
        **defaults: Keyword arguments passed to every constructor.

    Returns:
        A ``{class_name: instance}`` dict, one entry per class.
    """
    return {clazz.__name__: clazz(**defaults) for clazz in clazzes}
