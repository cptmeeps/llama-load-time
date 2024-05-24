import os, math, time, random, json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
from PIL import Image
from sentencepiece import SentencePieceProcessor
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
from torch import nn, tensor


@dataclass
class ModelArgs:
  dim: int = 4096
  n_layers: int = 32
  n_heads: int = 32
  n_kv_heads: Optional[int] = None
  vocab_size: int = 32000
  multiple_of: int = 256  
  ffn_dim_multiplier: Optional[float] = None
  norm_eps: float = 1e-5
  max_batch_size: int = 32
  max_seq_len: int = 1024
  img_len: int = 577
  img_dim: int = 1024
  max_gen_len: int = 32

class Tokenizer:
  def __init__(self, model_path: str):
    assert os.path.isfile(model_path), model_path
    self.sp_model = SentencePieceProcessor(model_file=model_path)

    self.n_words: int = self.sp_model.vocab_size()
    self.bos_id: int = self.sp_model.bos_id()
    self.eos_id: int = self.sp_model.eos_id()
    self.pad_id: int = self.sp_model.pad_id()
    assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

  def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
    assert type(s) is str
    t = self.sp_model.encode(s)
    if bos:
      t = [self.bos_id] + t
    if eos:
      t = t + [self.eos_id]
    return t

  def decode(self, t: List[int]) -> str:
    return self.sp_model.decode(t)

# layers

class RMSNorm(torch.nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x):
    output = self._norm(x.float()).type_as(x)
    return output * self.weight

class MLP(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.w1 = nn.Linear(args.img_dim, args.dim, bias=True).to(dtype=torch.float16)
    self.gelu = nn.GELU()
    self.w2 = nn.Linear(args.dim, args.dim, bias=True).to(dtype=torch.float16)

  def forward(self, x):
    h = checkpoint(lambda x: self.gelu(self.w1(x)), x)
    h = self.w2(h)
    return h

class Attention(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.n_heads = args.n_heads
    self.head_dim = args.dim // args.n_heads
    self.wq = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
    self.wk = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
    self.wv = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
    self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)  

  def reshape_for_broadcast(self, freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

  def apply_rotary_emb(self, xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

  def forward(self, x, freqs_cis, mask=None):
    bsz, seqlen, _ = x.shape
    # apply linear layer with checkpointing
    xq = checkpoint(lambda x: self.wq(x), x)
    xk = checkpoint(lambda x: self.wk(x), x)
    xv = checkpoint(lambda x: self.wv(x), x)
    # reshape to heads and head dims
    xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
    # apply rotary encoding
    xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    # transpose heads to 2D for easier computation
    xq = xq.transpose(1, 2)
    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2)
    # dot product of q v, scaled by the sqrt of head dims
    dot_product = torch.matmul(xq, xk.transpose(2, 3))
    scores = dot_product / math.sqrt(self.head_dim)
    # apply masks
    if mask is not None:
      scores = scores + mask
    # softmax to get attn scores
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    # get weighted sum from scores and value
    output = torch.matmul(scores, xv)
    # reshape back to original shape, apply linear layer
    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    return self.wo(output)

class FeedForward(nn.Module):
  def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier=None):
    super().__init__()
    hidden_dim = int(2 * hidden_dim / 3)
    if ffn_dim_multiplier is not None:
      hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    self.w1 = nn.Linear(dim, hidden_dim, bias=False)
    self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    self.w3 = nn.Linear(dim, hidden_dim, bias=False)

  def forward(self, x):
    x = checkpoint(lambda x: F.silu(self.w1(x)) * self.w3(x), x)
    return self.w2(x)

class TransformerBlock(nn.Module):
  def __init__(self, layer_id: int, args):
    super().__init__()
    self.n_heads = args.n_heads
    self.dim = args.dim
    self.head_dim = args.dim // args.n_heads
    self.attention = Attention(args)
    self.feed_forward = FeedForward(
      dim=args.dim,
      hidden_dim=4 * args.dim,
      multiple_of=args.multiple_of,
      ffn_dim_multiplier=args.ffn_dim_multiplier,
    )
    self.layer_id = layer_id
    self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
    self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

  def forward(self, x, freqs_cis, mask=None, ):
    h = x + self.attention.forward(
      self.attention_norm(x), freqs_cis, mask
    )
    out = h + self.feed_forward.forward(self.ffn_norm(h))
    return out

class Transformer(nn.Module):
  def __init__(self, params):
    super().__init__()
    self.params = params
    self.vocab_size = params.vocab_size
    self.n_layers = params.n_layers
    self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
    self.layers = torch.nn.ModuleList()
    self.freqs_cis = self.precompute_freqs_cis(
      self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
    )
    
    for layer_id in range(params.n_layers):
      self.layers.append(TransformerBlock(layer_id, params))
    self.norm = RMSNorm(params.dim, eps=params.norm_eps)
    self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

  def precompute_freqs_cis(self, dim, end, theta=10000.0):
    indices = torch.arange(0, dim, 2) 
    sliced_indices = indices[: (dim // 2)].float()
    scaled_indices = sliced_indices / dim
    theta_power = theta ** scaled_indices
    freqs = 1.0 / theta_power
    t = torch.arange(end, device="cuda")
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

  # @torch.inference_mode()
  def forward(self, toks): 
    _bsz, seqlen = toks.shape
    h = self.tok_embeddings(toks)
    
    self.freqs_cis = self.freqs_cis.to(h.device)
    freqs_cis = self.freqs_cis[:seqlen]
    mask = None
    
    if seqlen > 1:
      mask = torch.full((seqlen, seqlen), float("-inf"), device=toks.device)
      mask = torch.triu(mask, diagonal=1)
      mask = torch.hstack([
        torch.zeros((seqlen, 0), device=toks.device), mask
      ]).type_as(h)
    
    for layer in self.layers:
      h = layer(h, freqs_cis, mask)
    h = self.norm(h)
    output = self.output(h).float()
    return output

# run model

def build(model_args):
  print('building llama')
  start_time = time.time()
  torch.cuda.set_device(0)
  torch.set_default_tensor_type(torch.cuda.HalfTensor)  
  
  ckpt = torch.load("consolidated.00.pth", map_location="cuda")
  model = Transformer(model_args)
  model.load_state_dict(ckpt, strict=False)
  model = model

  print(f"loaded in {time.time() - start_time:.2f} seconds")
  print(f"alloc: {torch.cuda.memory_allocated('cuda')/1024**3:.1f}")
  print(f"cached: {torch.cuda.memory_reserved('cuda')/1024**3:.1f}")

  return model

def generate(model, tokenizer, model_args, prompt_toks, max_gen=32):
  start_time = time.time()
  bsz = len(prompt_toks)
  if max_gen:
    max_gen_len = max_gen
  else:  
    max_gen_len = model_args.max_seq_len - 1
  min_tok_len = min(len(t) for t in prompt_toks)
  max_tok_len = max(len(t) for t in prompt_toks)
  ttl_len = min(model_args.max_seq_len, max_gen_len + max_tok_len)

  pad_id = tokenizer.pad_id
  gen_toks = torch.full((bsz, ttl_len), pad_id, dtype=torch.long)
  for k, t in enumerate(prompt_toks):
    gen_toks[k, : len(t)] = torch.tensor(t, dtype=torch.long)
  eos_reached = torch.tensor([False] * bsz)
  input_text_mask = gen_toks != pad_id
  
  for cur_pos in range(min_tok_len, ttl_len):
    logits = model.forward(gen_toks[:, :cur_pos])
    nxt_tok = torch.argmax(logits[:, -1], dim=-1)
    nxt_tok = nxt_tok.reshape(-1)
    nxt_tok = torch.where(input_text_mask[:, cur_pos], gen_toks[:, cur_pos], nxt_tok)
    gen_toks[:, cur_pos] = nxt_tok
    eos_reached |= (~input_text_mask[:, cur_pos]) & (nxt_tok == tokenizer.eos_id)
    if all(eos_reached): break

  out_tkns = []
  for i, toks in enumerate(gen_toks.tolist()):
    prompt_toks_len = len(prompt_toks[i])
    toks = toks[prompt_toks_len : prompt_toks_len + max_gen_len]
    if tokenizer.eos_id in toks:
      eos_idx = toks.index(tokenizer.eos_id)
      toks = toks[:eos_idx]
    out_tkns.append(toks)  
  print(f"generated in {time.time() - start_time:.2f} seconds")
  return out_tkns
