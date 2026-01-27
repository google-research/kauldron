# KTyping Style Guide

This is a style guide for `ktyping`, a library for runtime shape and dtype
checking of arrays in Python.

## General Guidance

*   **Prioritize readability**: If an annotation becomes too complex, consider
    whether the check can be simplified or if a less stringent check would
    suffice.

*   **Be consistent**: Use consistent naming and annotation styles within your
    project.

*   **Be specific when possible**: If you know the exact dtype or shape,
    specify it.

## Import
Directly import types such as `Float` or `UInt8` from `kauldron.ktyping`
(similar to imports from `typing`).
This is recommended for readability of the annotations.

For other symbols (e.g. `typechecked`, `check_type`, `shape`, `dim`, ...)
you can either add them to the direct imports or use them via
`import kauldron.ktyping as kt` as you prefer. For example:

```python {.good}
import kauldron.ktyping as kt
from kauldron.ktyping import Float, UInt8  # pylint: disable=g-multiple-import,g-importing-member

@kt.typechecked
def fn(x: Float["b n c"]) -> UInt8["n n"]:
  n = kt.dim["n"]
  return np.zeros((n, n), dtype=np.uint8)
```

```python {.bad}
import kauldron.ktyping as kt

@kt.typechecked
def fn(x: kt.Float["b n c"]) -> kt.UInt8["n n"]:
  n = kt.dim["n"]
  return np.zeros((n, n), dtype=np.uint8)
```

## Annotations

Use `ktyping`'s array types to annotate arrays to enable runtime checks for the
array class (e.g. `jax.Array`, `np.ndarray`), the `dtype`, and the `shape`.
Annotations in ktyping use the form `Float["b d"]`, where the string part is the
shape specification. Note that this is unlike the long-form annotations of
`jaxtyping` which would look like `Float[Array, "b d"]`.

*   Use `Array` if the dtype does not matter, and `Num` for any numerical type.
*   Use `Float`, `Int`, `Bool`, `Complex` for general-purpose float, integer,
    boolean or complex arrays.
*   Use more specific types like `UInt8`, `Float32`, `BFloat16` when the exact dtype is important.
*   Use `Scalar`, `ScalarInt`, `ScalarFloat`, `ScalarBool` for arguments that
    support 0-dimensional arrays and Python scalar values.
*   Use `PRNGKey` for JAX random keys.
*   For tensorflow array, use the array types with `Tf...` prefix (e.g. `TfFloat`).
*   Use the array types with the `X...` prefix (e.g. `XFloat`) when you want to
    support JAX, Numpy, TensorFlow, and Torch arrays simultaneously.

### Naming Conventions for Dimensions

Consistent naming of dimensions helps readability and allows `ktyping` to
correctly match dimensions across different arguments.

*   Use single lowercase letters for common dimensions
    (but only if they are clear from context):
    *   `b` / `*b`: batch dimension(s)
    *   `n`: number of items (e.g., points in a point cloud, tokens in a sequence)
    *   `h`, `w`: height and width for images
    *   `c`: channels or features
    *   `d`: embedding dimension
    *   `t`: time or sequence length
*   Use descriptive lowercase names for other dimensions (e.g., `num_tokens`,
    `embedding_dim`).
*   For dimensions that do not matter (e.g. if the function only operates on the
    last dimension) consider using `...` or `*any`.
    The latter is preferred if multiple arguments or the return value share the
    same set of dimensions.
*   Prefixing with `_` like `_b` makes a dimension anonymous. Its value is
    checked but not remembered. Use `...` as a shortcut for `*_`.

## Runtime typechecking
To check the type annotation during runtime, `ktyping` offers the `typechecked`
decorator / context, and the `check_type` function.

### Functions
Use the `@kt.typechecked` decorator for functions whenever the arguments contain
arrays and have shape annotations so that the annotations are checked at
runtime.
The only exception should be for functions that would be slowed down too much by
the small overhead of `@typechecked`.
NOTE: This is generally never the case for any jax jitted function, since these
are only executed once during tracing, so the overhead only affects compilation.

### Dataclasses
Decorating dataclasses with `@kt.typechecked`can be useful when they are mainly
containers for a set of related arrays. E.g.:

```python {.good}
@kt.typechecked
@dataclasses.dataclass(kw_only=True, frozen=True)
class Example:
  image: Float["*b h w 3"]
  depth: Float["*b h w 1"]
  label: Int["*b 1"]
```

The typechecked decorator should generally be avoided for dataclasses that
contain many non-array fields, or any complex logic, because these cases are
more likely to clash with the typechecking logic.

### TypedDicts
For complicated arguments or return types it is often better to define a
`TypedDict` instead of an overly complicated or overly generic
`dict[str, Array['...']]` annotation:

```python {.good}
import typing
from kauldron.ktyping import Bool, Float, typechecked

class ModelInput(typing.TypedDict):
  tokens: Float["*b n d"]
  labels: Int["*b"]
  mask: Bool["*b n"]

def compute_loss(inp: ModelInput) -> ScalarFloat:
  ...
```

```python {.bad}
def compute_loss(inp: Array['*b ...']) -> ScalarFloat:
  ...
```

```python {.bad}
def compute_loss(inp: Float["*b n d"] | Int["*b"] | Bool["*b n"]) -> ScalarFloat:
  ...
```

Note: You might also consider using a dataclass for these cases.

### Context
Using `kt.typechecked` as a context manager can be useful to avoid
clashes of dimensions.
For example in the case of named aliases for array types, which would otherwise
have to be spelled out with differently named dimensions.
But this feature should be used sparingly.

```python
Image = Float["b h w 3"]

def foo(x: Image):
  y = resize(x, width=kt.dim['w']//2)
  # kt.check_type(y, Image)  # ERROR, because w has changed.
  with kt.typechecked():
    kt.check_type(y, Image)  # OK, because the dims here are separate

```

### `check_type`
It is recommended, especially for long functions, to add additional runtime
checks to intermediate variables via `kt.check_type`.
This often helps verify assumptions about the shapes, and improves clarity.

```python
@kt.typechecked
def forward(self, img: Float["*b h w c"]) -> Float["*b {self.num_classes}"]:
  tokens = self.tokenize(img)
  check_type(tokens, Float["*b n d"])
  for l in self.layers:
    tokens = l(tokens)

  pre_logits = self.mean_average_pool(tokens)
  check_type(pre_logits, Float["*b d"])

  logits = nn.Dense(features=self.num_classes)(final)
  return logits
```

## Batch shapes, Broadcasting, Indexing

All models and functions should ideally support arbitrary batch shapes, i.e.
`"*b h w c"` rather than `"b h w c"`.
Note that this implies indexing should always be used from the right, i.e. in
general, only negative indexing should be used, even if positive indexing would
be an option.

``` python {.good}
def center(img: Float["*b h w c"]) -> Float["*b h w c"]:
  return img - img.mean((-3, -2, -1))
```

``` python {.bad}
def center(img: Float["b h w c"]) -> Float["b h w c"]:
  return img - img.mean((1, 2, 3))
```

Arrays that are frequently combined / used together (e.g. logits, labels and
masks) should generally have the same number of dimensions to allow
implicit broadcasting. That means their shapes should be right-padded with 1s:

``` python {.good}
def ce(logits: Float["*b d"], labels: Int["*b 1"], mask: Bool["*#b 1"]) -> FloatScalar:
  ...
```

``` python {.bad}
def ce(logits: Float["*b d"], labels: Int["*b"], mask: Bool["*b"]) -> FloatScalar:
  ...
```

This allows all implementations to handle arbitrary batch dimensions out
of the box without the need for explicit reshapes or vmaps. It also allows many
operations to generalize e.g. from images to video, or from single-view to
multi-view.

## Accessing Dimension Values

Inside a `@typechecked` function or context, use `kt.dim` to interact with the
inferred dimensions:

```python {.good}
@kt.typechecked
def my_func(x: Float["*b n d"]):
  batch_size = kt.dim['*b']  # get the value of "*b"
  kt.dim['h'] = 64  # set the value of "h" for later use
  ...
```

To construct shapes the `shape` function can be used, and is generally more
readable than manually constructing tuples from other shapes:

```python {.good}
@kt.typechecked
def my_func(tokens: Float["*b n d"]):
  att_mask = jnp.zeros(kt.shape("*b n n"))
  ...
```

```python {.bad}
@kt.typechecked
def my_func(tokens: Float["*b n d"]):
  att_mask = jnp.zeros(tokens.shape[:-2] + (tokens.shape[-2], tokens.shape[-2]))
  ...
```