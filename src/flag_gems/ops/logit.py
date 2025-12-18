import logging
from typing import Optional

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def logit_forward_kernel(x, eps):
    z = tl.minimum(tl.maximum(x.to(tl.float32), eps), 1.0 - eps)
    return tl.log(z / (1.0 - z))


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def logit_backward_kernel(x, dy, eps):
    x_f32 = x.to(tl.float32)
    is_clamped = False
    if eps >= 0.0:
        is_clamped = (x_f32 < eps) or (x_f32 > 1.0 - eps)
        x_f32 = tl.minimum(tl.maximum(x_f32, eps), 1.0 - eps)
    dx = dy.to(tl.float32) / (x_f32 * (1.0 - x_f32))
    return tl.where(is_clamped, 0.0, dx)


def logit(self, eps: Optional[float] = None):
    logger.debug("GEMS LOGIT FORWARD")
    if eps is None:
        eps = -1.0
    output = logit_forward_kernel(self, eps)
    return output


def logit_backward(grad_output, self, eps: Optional[float] = None):
    logger.debug("GEMS LOGIT BACKWARD")
    if eps is None:
        eps = -1.0
    grad_input = logit_backward_kernel(self, grad_output, eps)
    return grad_input


def logit_(A, eps: Optional[float] = None):
    logger.debug("GEMS LOGIT_ FORWARD")
    if eps is None:
        eps = -1.0
    out = logit_forward_kernel(A, eps, out0=A)
    return out


def logit_out(A, out, eps: Optional[float] = None):
    logger.debug("GEMS LOGIT_OUT FORWARD")
    if eps is None:
        eps = -1.0
    out = logit_forward_kernel(A, eps, out0=out)
    return out
