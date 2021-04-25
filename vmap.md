# Why vmap isn't magic (and how to implement it)
The Sapir-Worf hypothesis is the idea that the structure of our language has an influence on the structure of our thoughts. In computer science, there's a corollary

> the structure of our language/framework has an influence on the structure of our thoughts.

We've seen this play out first-hand in machine learning. How essential is autograd to the current state of our field? If autograd did not exist, how would that impact our research?

The cool thing about autograd is that at its core, it's not really that complicated! Numerous people (Karpathy, me, etc.) have built out miniature autograd systems, and numerous machine learning courses teach that "backpropagation" is just "the chain rule + dynamic programming".

In recent years, there's been another innovation in this space of transformations - vmap. Vmap is a function transformation (pioneered by Jax) that stands for "vectorizing map". Semantically, it takes in a function `f(x)` and adds a new dimension that it maps over. It's equivalent to this:

```python
def vmap(f):
    def vectorized_f(x):
        return torch.stack([f(x[i]) for i in range(x.shape[0])])
    return vectorized_f
```

But wait, this code is awful! It violates "How to Make Your Deep Learning Code Run Fast 101" - never use Python loops. This gets to the real magic of vmap: somehow, it makes your code run as fast as if you'd batched it yourself.

There are 2 common misconceptions about vmap that I'd like to clear up.

1. **It requires a codegen compiler.** The description of vmap as an "vectorizing map" leads many to assume that it works similarly to an "auto-vectorizing" compiler (i.e. a compiler that automatically inserts SIMD instructions). **This is not true!** Vmap can be written purely as a "source to source" transform - you can write a vmap that takes in Python code and outputs other Python code. In other words, vmap is completely independent of XLA, and can be implemented on top of any other array-processing framework, like PyTorch or Numpy.
2. **Vmap is dark magic that's hard to understand.** Vmap doesn't require any deep compiler or codegen understanding. In fact, I would argue that vmap is actually much simpler than autograd. The goal of this post is to demonstrate how you could have implemented vmap yourself.

## Why Should You Care?
One natural question is: Why should I care about vmap? TODO

## How to Implement Vmap
Before we delve into how vmap "automatically" batches your function, let's run through what batching is, and some examples about what it takes to manually batch your code.


### What is ... Batching?
Let's say you have function $f(x)$ that takes in one element and returns one element. In terms of shape notation, it maps from $() \to ()$ (i.e. a zero-dimension tensor to a zero-dimension tensor).

What happens after you batch it? Well, since you're adding a new batch dimension $B$, it now maps from $(B,) \to (B,)$ (i.e. a one-dimensional tensor (a vector) to a one-dimensional tensor).

The same thing applies for a function $f(x)$ that takes in a tensor of shape $(S_0, S_1, ...)$ and outputs a tensor of shape $(T_0, T_1, ...)$. Adding a new batching dimension turns it into a function that takes in a tensor of shape $(B, S_0, S_1, ...)$ and outputs a tensor of shape $(B, T_0, T_1, ...)$.

Now, let's see what it takes to manually batch a couple of example functions.

### Pointwise Operations
```python
def f(x: Tensor[3,5,10]) -> Tensor[3,5,10]:
    return x * 2 + 5
```
<details>
<summary>How to Batch?</summary>

Let's say that the original shape of the input tensor was $(3,5,10)$, applies some point-wise operations, and returns a tensor of shape $(3,5,10)$. What do we need to do in order to get the function to work on a shape of $(B, 3, 5, 10)$?

Well, trick question - this function already works on arbitrary input shapes! So we don't actually need to do anything when it's a simple pointwise operation.
</details>


### Dimension-Specific Operations
```python
def f(x: Tensor[3,5,10]) -> Tensor[3,10]:
    return x.sum(dim=1)
```
<details>
<summary>How to Batch?</summary>

Let's say that this takes in a tensor with shape $(3,5,10)$, sums the first dimension, and returns a tensor of shape $(3,10)$. The primary complication now is that we're summing over the first dimension, so our previous batching strategy of "do nothing" isn't going to work.

First of all, what should the intended output shape be? It's quite simple - we just add a $B$ to the output shape, to get $(B,3,10)$.

On the other hand, the code requires a little more work to translate. Since we're adding a new batch dimension in front, the dimension we previously pointed at is now the second dimension. Note that if we were using named axes, this wouldn't be a problem at all.

```python
def f(x: Tensor[B,3,5,10]) -> Tensor[B,3,10]:
    return x.sum(dim=2)
```
</details>

### Batching Operations that are Already Batched
```python
def f(a: Tensor[N, C, H, W]) -> Tensor[N, C, H, W]:
    return self.conv2d(a)
```
<details>
<summary>How to Batch?</summary>

If our operation already supports batches, what should we do? This is one of the easiest cases - we can simply squeeze our batch dimensions together and then unsqueeze them afterwards.

```python
def f(a: Tensor[B, N, C, H, W]) -> Tensor[B, N, C1, H1, W1]:
    squeezed_a: Tensor[B*N, C, H, W] = a.reshape(B*N, -1)
    out: Tensor[B*N, C1, H1, W1] = self.conv2d(squeezed_a)
    return out.reshape(B, N, -1)
```
</details>

### Operations That Don't Accept an Arbitrary Rank
```python
def f(a: Tensor[5], b: Tensor[5]) -> Tensor[]:
    return torch.dot(a, b)
```
<details>
<summary>How to Batch?</summary>

Now, our function takes in two vectors with shape $(5)$ and outputs a 0-dim tensor. The primary complication is that `torch.dot` explicitly only takes tensors with 1 dimension. So, if `a` or `b` get a new batch dimension, no amount of massaging the arguments is going to allow `torch.dot` to accept the new shape.

In order to get this to work, we're going to need to do some thinking. Let's assume that we only want to batch `a` - that is, the signature of `f` is now `f(Tensor[B, 5], Tensor[5]) -> Tensor[B]`.

If we think about it a little, we can see that this is identical to matrix-vector multiplication (at least, the shapes match up). So, we can simply substitute our `torch.dot` with a `torch.mv`!

```python
def f(a: Tensor[B, 5], b: Tensor[5]) -> Tensor[B]:
    return torch.mv(a, b)
```
</details>


### Broadcasting Operations
```python
def f(a: Tensor[], b: Tensor[3]) -> Tensor[3]:
    return a*b
```
<details>
<summary>How to Batch?</summary>

Now, let's say that you're multiplying a scalar by a vector with shape $(3)$, resulting in a vector with shape $(3)$. Now, we'd like to batch $a$. That is, we'd like to multiply a vector with shape $(B)$ by a vector with shape $(3)$ to get a matrix with shape $(B, 3)$. In other words, an outer product.

Unlike the previous point-wise operations, there's another wrinkle here: broadcasting. The multiplication that's currently going on is between tensors of shape $()$ and $(3)$, but the broadcasting logic implicitly converts it to a multiplication between tensors of shape $(1)$ and $(3)$.

Now, we can see that the actual shapes of our batched inputs are $(B,1)$ and $(3)$. Thus, we simply need to reshape our inputs to enable the desired broadcasting.

```python
def f(a: Tensor[B], b: Tensor[3]) -> Tensor[B, 3]:
    a = a.reshape(B, 1)
    b = b.reshape(1, 3)
    return a*b
```
</details>

### Control Flow
```python
def f(a: Tensor[5]) -> Tensor[5]:
    if a.sum(dim=0) > 0:
        return a
    else:
        return a*2
```
<details>
<summary>How to Batch?</summary>

Finally, how do we batch over (limited) conditionals? There are a couple different ways to do so, but one way is to simply execute both sides of the conditional for each element of your batch, and then select from the results. For example, we could translate the above control flow into
```python
def f(a: Tensor[B, 5]) -> Tensor[B, 5]:
    cond: Tensor[B] = (a.sum(dim=1) > 0).unsqueeze(1)
    true_result: Tensor[B, 5] = a
    false_result: Tensor[B, 5] = a*2
    return torch.where(cond, true_result, false_result)
```
Unfortunately, this does redundant computation. Another way this could be implemented is by separating out the "true" part of the batch from the "false" part, executing them separately, and then stitching them back together.
```python
def f(a: Tensor[B, 5]) -> Tensor[B, 5]:
    cond: Tensor[B] = (a.sum(dim=1) > 0)
    true_idxs = cond.nonzero().squeeze(1)
    false_idxs = (cond != 0).nonzero().squeeze(1)
    true_batch = a[true_idxs]
    false_batch = a[false_idxs]
    true_result = true_batch
    false_result = false_batch*2
    result = torch.zeros(a.shape)
    result[true_idxs] = true_batch
    result[false_idxs] = false_batch
    return result
```
Although this has the advantage of not performing redundant computation, it comes at the cost of being far more dynamic. Whether this is faster or not probably depends on what you're computing.
</details>

## An Example Vmap Implementation
Now, let's walk through an example vmap implementation. I'll be implementing this in PyTorch FX, which is a framework for writing function transformations on top of PyTorch. The implementation should be fairly similar if you write it on top of jaxprs. However, I'm scared of jaxprs and FX has the advantage that it's a Python to Python transformation, which I think will help clarify what's happening. Also, I already did it in FX so I have the code lying around. However, much of what I write is based off Jax's [non-kiddy implementation of vmap](https://github.com/google/jax/blob/master/jax/interpreters/batching.py), so check that out!

### High Level Approach
In the previous section, we wrote about how to batch individual operations. However, how do we batch an entire function? How do things that aren't batched and batched interact?

The general strategy is to augment each tensor with a new "batch dimension", and execute our program as normal. Then, for each operation, we:

1. If all input tensors don't have a batch dimension, then we execute that operation normally.
2. On the other hand, if any of the tensors have a batch dimension, that implies that the output will have a batch dimension as well, and we need to follow one of the above rules.

Now, given the above, we can implement "batching rules" for each operation. For example, let's take the squeeze batching rule (lifted near directly from [Jax!](https://github.com/google/jax/blob/555aba891d8c1cf8db095e9b85b9d5f50597f840/jax/_src/lax/lax.py#L3609)).

```python
def squeeze_batching_rule(x, dim):
    x = x.movedim(x.bdim, 0)
    if dim >= 0:
        return torch.squeeze(x, dim + 1) # to adjust for the new bdim
    else:
        return torch.squeeze(x)
```
Although, in PyTorch, this could be implemented at the dispatcher (like the actual PyTorch vmap [does it](https://pytorch.org/docs/master/generated/torch.vmap.html)) or as a special [Tensor type](https://github.com/pytorch/pytorch/pull/32836), doing it with FX allows us to pull out the resulting Python code.

### PyTorch FX Implementation
The first thing we need to with FX is to capture a symbolic representation of our function [insert FX link here](). Next, as FX does not capture shapes by default, we will run a shape propagation pass.

```python
import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx import Proxy

from torch.fx.passes.shape_prop import ShapeProp

example_args = (torch.randn(()), torch.randn(5))
def f(x, y): # We will rewrite this function into batched outer product.
    return x*y

fx_model = fx.symbolic_trace(f)
ShapeProp(fx_model).propagate(*example_args)
print(fx_model.code)

>>> def forward(self, x, y):
>>>     mul = x * y
>>>     return mul
```

When re-interpreting our code, we will construct a new graph and a new environment, so we will set up some utility functions here.
```python
new_graph: fx.Graph = fx.Graph()
env = {}
def lookup_env(l):
    return fx.node.map_aggregate(l, lambda x: env[x.name] if isinstance(x, fx.Node) else x)
```

Next, we'll iterate over our graph, applying our batching rule when relevant. For this example, our batching rule will only work with functions (so no modules or methods). For now, assume that `apply_batching_rule`
```python
for node in fx_model.graph.nodes:
    if node.op == 'placeholder':
        # If the node is an input placeholder, we simply copy it over and
        # annotate it with the batch dimension from `in_axes`.
        new_node = new_graph.placeholder(node.name)
        new_node.bdim = next(in_axes)
        new_node.shape = node.shape
        env[node.name] = new_node
    elif node.op == 'output':
        new_graph.output(env[node.args[0].name])
    elif node.op == 'call_function':
        new_args = lookup_env(node.args)
        # If any of the inputs to the function has a new batch dimension,
        # we will need to use our batching rules. Otherwise, we will simply
        # copy the node over.
        if any([x.bdim is not None for x in new_args if isinstance(x, fx.Node)]):
            new_node = apply_batching_rule(node.target, *new_args)
        else:
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            new_node.bdim = None
        new_node.shape = node.shape
        env[node.name] = new_node
    else:
        raise RuntimeError("Not yet implemented")
```


