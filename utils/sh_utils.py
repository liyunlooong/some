#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def RGB2SH(rgb):
    return rgb / C0


def SH2RGB(sh):
    return sh * C0

@torch.jit.script
def eval_sh(deg: int, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = 0.28209479177387814 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                0.4886025119029199 * y * sh[..., 1] +
                0.4886025119029199 * z * sh[..., 2] -
                0.4886025119029199 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    1.0925484305920792 * xy * sh[..., 4] +
                    -1.0925484305920792 * yz * sh[..., 5] +
                    0.31539156525252005 * (2.0 * zz - xx - yy) * sh[..., 6] +
                    -1.0925484305920792 * xz * sh[..., 7] +
                    0.5462742152960396 * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                -0.5900435899266435 * y * (3 * xx - yy) * sh[..., 9] +
                2.890611442640554 * xy * z * sh[..., 10] +
                -0.4570457994644658 * y * (4 * zz - xx - yy)* sh[..., 11] +
                0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                -0.4570457994644658 * x * (4 * zz - xx - yy) * sh[..., 13] +
                1.445305721320277 * z * (xx - yy) * sh[..., 14] +
                -0.5900435899266435 * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + 2.5033429417967046 * xy * (xx - yy) * sh[..., 16] +
                            -1.7701307697799304 * yz * (3 * xx - yy) * sh[..., 17] +
                            0.9461746957575601 * xy * (7 * zz - 1) * sh[..., 18] +
                            -0.6690465435572892 * yz * (7 * zz - 3) * sh[..., 19] +
                            0.10578554691520431 * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            -0.6690465435572892 * xz * (7 * zz - 3) * sh[..., 21] +
                            0.47308734787878004 * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            -1.7701307697799304 * xz * (xx - 3 * yy) * sh[..., 23] +
                            0.6258357354491761 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result
