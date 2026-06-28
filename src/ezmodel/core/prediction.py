"""Named result type for model predictions: the mean plus optional extras."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Prediction:
    """The result of ``Model.predict``: the mean plus any requested extras, named.

    ``y`` is always populated; ``sigma`` and ``grad`` are ``None`` unless their flag was
    set on the call, so callers read fields instead of guessing tuple positions.

    Attributes:
        y: Predicted mean, shape ``(m, q)``.
        sigma: Predictive standard deviation, shape ``(m, 1)``, or ``None`` when not requested.
        grad: Gradient of the mean w.r.t. the query point, shape ``(m, d)``, or ``None``.
    """

    y: np.ndarray
    sigma: Optional[np.ndarray] = None
    grad: Optional[np.ndarray] = None
