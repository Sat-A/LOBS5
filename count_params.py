#!/usr/bin/env python3
import sys
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5')

import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
from lob.lob_seq_model import PaddedLobPredModel
from lob.encoding import Vocab

# 配置 (与 run_lobster_padded_large.sh 一致)
d_model = 3072
n_layers = 32
n_message_layers = 2
n_book_pre_layers = 1
n_book_post_layers = 1
ssm_size_base = 3072
blocks = 48
d_book = 503
conj_sym = True
vocab_size = len(Vocab())  # 2112

# === 正确的 SSM 初始化 (来自 init_train.py:204-225) ===
ssm_size = ssm_size_base
block_size = int(ssm_size / blocks)  # 64
Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

if conj_sym:
    block_size = block_size // 2  # 32
    ssm_size = ssm_size // 2      # 1536

Lambda = Lambda[:block_size]
V = V[:, :block_size]
Vc = V.conj().T

Lambda = (Lambda * jnp.ones((blocks, block_size))).ravel()
V = block_diag(*([V] * blocks))
Vinv = block_diag(*([Vc] * blocks))

print(f"SSM init: block_size={block_size}, ssm_size={ssm_size}, blocks={blocks}")
print(f"Lambda shape: {Lambda.shape}")
print(f"V shape: {V.shape}")
print(f"Vinv shape: {Vinv.shape}")

ssm_init_fn = init_S5SSM(
    H=d_model, P=ssm_size,
    Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag,
    V=V, Vinv=Vinv,
    C_init="trunc_standard_normal",
    discretization="zoh", dt_min=0.001, dt_max=0.1,
    conj_sym=conj_sym, clip_eigs=True, bidirectional=False,
)

model = PaddedLobPredModel(
    ssm=ssm_init_fn, d_output=vocab_size,
    d_model=d_model, d_book=d_book,
    n_message_layers=n_message_layers, n_fused_layers=n_layers,
    n_book_pre_layers=n_book_pre_layers, n_book_post_layers=n_book_post_layers,
    activation="half_glu1", dropout=0.0, training=True, mode="none",
    prenorm=True, batchnorm=False, bn_momentum=0.95, step_rescale=1.0,
)

# 初始化
print("Initializing model...")
rng = jax.random.PRNGKey(42)
variables = model.init(
    {"params": rng, "dropout": rng},
    jnp.ones((100,), dtype=jnp.int32), jnp.ones((100, d_book)),
    jnp.ones((100,)), jnp.ones((100,)),
    method='__call_ar__'
)
params = variables["params"]
print("Model initialized!")

# 统计
def count_params(pytree):
    return sum(x.size * (2 if x.dtype in [jnp.complex64, jnp.complex128] else 1)
               for x in jax.tree_util.tree_leaves(pytree))

total = count_params(params)
print(f"\n{'='*70}")
print(f"实测参数分布 (d_model={d_model}, n_layers={n_layers}, P={ssm_size})")
print(f"{'='*70}\n")

components = {}
if 'message_encoder' in params:
    components['Message Encoder'] = count_params(params['message_encoder'])
if 'book_encoder' in params:
    be = params['book_encoder']
    components['Book Pre-layers'] = sum(count_params(be[k]) for k in be if k.startswith('pre_layers'))
    if 'projection' in be:
        components['Book Projection'] = count_params(be['projection'])
    components['Book Post-layers'] = sum(count_params(be[k]) for k in be if k.startswith('post_layers'))
if 'fused_s5' in params:
    components['Fused S5 (32层)'] = count_params(params['fused_s5'])
if 'decoder' in params:
    components['Decoder'] = count_params(params['decoder'])

for name, cnt in sorted(components.items(), key=lambda x: -x[1]):
    pct = cnt / total * 100
    bar = "█" * int(50 * cnt / max(components.values()))
    print(f"  {name:25s} {cnt:>15,}  ({pct:6.2f}%)  {bar}")

print(f"\n{'-'*70}")
print(f"  {'TOTAL':25s} {total:>15,}  (100.00%)")
print(f"\n  日志值: 1,028,249,709")
print(f"  实测值: {total:,}")
