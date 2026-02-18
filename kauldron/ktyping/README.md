# KTyping

`ktyping` is a small library for annotating and runtime checking of array
types including dtypes and shapes. It is an evolution of `jaxtyping`, and
addresses some of its problems (see
[differences with jaxtyping](#differences-with-jaxtyping)).

Basic usage:

```python
import jax.numpy as jnp
import jax.nn as nn
import kauldron.ktyping as kt
from kauldron.ktyping import Float

@kt.typechecked
def simple_attention(
    query: Float["*b n d"],
    key: Float["*b m d"],
    value: Float["*b m dv"],
    mask: Bool["*b n m"]
) -> Float["*b n dv"]:
  attn_weights = jnp.einsum("...nd,...md->...nm", query, key)
  attn_weights = nn.softmax(attn_weights, axis=-1)
  kt.check_type(attn_weights, Float["*b n m"])
  return jnp.einsum("...nm,...mv->...nv", attn_weights*mask, value)
```

## Shape Spec Mini-Language
`ktyping` uses a concise mini-language for shape specifications within strings.
A few illustrative examples (see below for explanations):

* `*b h w c`
* `batch time? hidden`
* `n h w 3|1`
* `h*w (c+1)//2`

The shape spec string in annotations like `Float["*b n d+1"]` is a list of
dimension expressions separated by spaces. Each dim expression can be:

* **Named dimensions**: `b h w c` (each matches exactly one axis, value is remembered)
* **Fixed dimensions**: `1024 3` (each matches exactly one axis of the given size)
* **Variable dimensions**:
  *   `*b`: Matches 0 or more dimensions
  *   `+d`: Matches 1 or more dimensions
  *   `t?`: Matches 0 or 1 dimension. If the dimension is absent, it is not remembered.
* **Anonymous dimensions**: dimensions whose name starts with an `_`
  * `_name`: Matches exactly one dimension, value is **not** remembered.
  * `...`: Matches 0 or more dimensions, values are **not** remembered. (Alias for `*_`).
* **Broadcastable dimensions**:
  *   `#c`: Matches a dimension named `c`, but also allows a dimension of size 1 to be broadcast.
  *   `*#b`: Matches 0 or more dimensions named `b`, allowing any of them to be 1.

* **Dimension choices**: `3|a`: Matches a dimension of size 3 OR `a`.
* **Expressions**: `d+1`, `h*w`, `(c+1)//2`: Python-like expressions are allowed but **cannot contain spaces**.
* **f-string expression**: `{features}`: Behaves as if the shape spec was
  evaluated as a Python f-string in the context of the function. For example:

  ```python
  @kt.typechecked
  def embed(x: Float["b n d"], features: int) -> Float["b n {features}"]:
    ...
  ```

NOTE: Note that (unlike in jaxtyping) you **can** use multiple variable dims in
  a single shape spec, e.g. `"*b *other c"` is valid. Keep in mind though, that
  this can lead to ambiguity in the assignments. E.g. passing an array of shape
  `(3, 7)` will have `c=7`, but is consistent with either `b=(), other=(3,)` or
  `b=(3,), other=()`. In general `ktyping` handles and -- if possible --
  resolves these ambiguities. But if you, for example, use `kt.shape("*b")` you
  might encounter an error if the ambiguity remained.

## Array Types
Common array types (accept both `numpy` and `jax` arrays):

*   **`Array`**: Generic array type, matches any array type.
*   **`Num`**: Any numerical type (`Int | Float | Complex`).
*   **`Bool`**: Boolean array.
*   **`Int`**: Any integer type (`SInt | UInt`).
*   **`SInt`**: Signed integer types. Subtypes: `Int8`, `Int16`, `Int32`, `Int64`.
*   **`UInt`**: Unsigned integer types. Subtypes: `UInt8`, `UInt16`, `UInt32`, `UInt64`.
*   **`Float`**: Any floating-point type. Subtypes: `BFloat16`, `Float16`, `Float32`, `Float64`.
*   **`Complex`**: Any complex type. Subtypes: `Complex64`, `Complex128`.
*   **`Scalar`**: A single number (int, float, complex, bool), either as a python built-in type or as a 0-dim array. Subtypes: `ScalarBool`, `ScalarInt`, `ScalarFloat`, `ScalarComplex`.
*   **`PRNGKey`**: A JAX PRNG key.

Prepend `Tf` to get **tensorflow** versions of the above. E.g. `TfInt64`.

Prepend `X` to get versions of the above that accept **any array class** (numpy,
jax, tensorflow, and torch). E.g. `XFloat32`.

You can use `|` for dtypes and dimensions. E.g. `(Float|UInt8)["b h w 3|4"]` as
shorthand for:

`Float["b h w 3"] | Float["b h w 4"] | UInt8["b h w 3"] | UInt8["b h w 4"]`

You can define **custom array types** as follows:

```python
from kauldron.ktyping import array_type_meta as atm, dtypes

VeryUsefulArray = atm.ArrayTypeMeta(
  "VeryUsefulArray",
  array_types=(atm.JaxArray, atm.TfArray), # accept jax and tf (but not numpy)
  dtype=dtypes.float32 | dtypes.bool)  # either float32 or bool

def foo(x: VeryUsefulArray["a b c"]):
  ...
```

## `kt.typechecked`
`kt.typechecked` enables runtime type checking (using `typeguard`).
It can be used as a decorator for functions and dataclasses, or as a
context manager.

### Function decorator

```python
@kt.typechecked
def fn(x: int, y: Float["a b"]) -> Bool["b"]:
  ...
```

Every time the function is called this:

* Opens a new [scope](#scopes)
* Type-Checks all the arguments
* Calls the function
* Type-Checks the return type
* Closes the scope

This also works for methods, `@classmethod`, `@staticmethod`, as well as
`@property`. It also offers limited support for generator functions,
where only the arguments and the return type are checked, but not the yield and
send types.

### Dataclass decorator

```python
@kt.typechecked
@dataclasses.dataclass
class PointCloud:
  pos: Float["n 3"]
  color: UInt8["n 3"]

# Fails because n from pos is different than n from color:
p = PointCloud(pos=np.zeros((17, 3)), color=np.ones((16, 3), dtype=np.uint8))
```

Decorating any `dataclass` with `@kt.typechecked` has two effects:

 1. Wraps the `__init__` function such that all the arguments are checked.
 2. Checks the attribute types when instances are passed as arguments to a
    `@kt.typechecked` function. Note that the arguments are checked in an
    isolated scope (named dims are not shared with the function scope).

### Context Manager
Manually open a new scope by using `kt.typechecked` as a context manager:

```python
with kt.typechecked():
  kt.check_type(x, Float["b n d"])
  kt.check_type(y, Float["b m d"])
  if isinstance(z, Int["n m"]):
    pass
```

### `kt.check_type`
To add additional typechecks (e.g. to the body of a function) use
`kt.check_type`:

```python
@kt.typechecked
def manual(x: Float["*b h w c"]):
  tok = encode(x)
  kt.check_type(tok, Float["*b n d"])

  out = kt.check_type(mean_avg_pool(tok), Float["*b d"])  # can also inline
  ...
```

Note that `kt.check_type` can only be called from within a `kt.typechecked`
function or context.

### `kt.isinstance`
`ktyping`'s custom `kt.isinstance` behaves like the built-in `isinstance` but it
fully supports:

 * ktyping annotations like `Float["b"]` (requires an active scope)
 * Parameterized types like `dict[str, int]` or `list[bool]`.
 * Union Types like `Union[byte, str]`, `int | float`, or `Optional[int]`

```python
@kt.typechecked
def fn(x: Float["*b h w c"]):
  if kt.isinstance(x, Float["1 h w 3"]):
    print("Found single image")
```
`kt.isinstance` behaves as follows:

  * It returns either `True` or `False`.
  * Unlike `kt.check_type` it **never modifies** the active scope.
  * It only requires an active scope when checking ktyping annotations. <br/>
    (i.e. `kt.isinstance(x, list[int] | int))` works even without an active scope)
  * Be aware that only the first element of container types are checked. <br/>
    (i.e. `kt.isinstance([1, "2"], list[int]) == True`).

Note: Python's built-in `isinstance(x, kt.Float["b"])` can also be used with
ktyping annotations, and works even without an active scope
(see [Working without an active scope](#working-without-an-active-scope)).
However, this is not recommended since it will silently reuse a parent scope
without raising an error
(e.g. when a `@kt.typechecked` decorator was forgotten).

## Scopes
Dimension assignments (e.g. `b=32`) are tracked using a stack of
`kt.ShapeScope`s. Shape checks always refer to the active (topmost) scope on the
stack (access via `kt.get_current_scope()`).

Scopes are opened and closed automatically for each `@kt.typechecked` function.
To avoid this (and reuse/modify the parent scope) use
`@kt.typechecked(new_scope=False)`.
You can also manually open a scope by using `kt.typechecked` as a context
manager.

```python
@kt.typechecked  # < Opens a new scope when fn is called
def fn(a: Float["a"]) -> Int["a"]:  # arguments checked against this scope
  b, c = something(a)
  # all checks inside reuse this scope (and modify it if they succeed).
  kt.check_type(b, Float["a b"])

  # isinstance checks against the active scope, but does not modify it.
  # So 'c' from Int["a c"] is not remembered (difference to jaxtyping).
  if isinstance(c, Int["a c"]):
    ...
    d = fn(a[:3])  # calling the function again opens (and closes) a new scope

  return e # The return type is checked and then the scope is closed
```

### Dataclasses and TypedDicts
If a dataclass is decorated with `@kt.typechecked`, a new scope is opened for
checking the attributes during its `__init__` method and when checking arguments
annotated as this dataclass.

```python
@kt.typechecked  # new scope opened used for checking all attributes
@dataclasses.dataclass
class Outputs:
  logits: Float["*b n d"]
  labels: Int["*b n 1"]
  attn: Float["*b n n"]

@kt.typechecked
def fn(net1: Outputs, net2: Outputs):  # net1 and net2 each get their own scope
  # so net1.logits.shape might differ from net2.logits.shape
  # Note that this is different from the behaviour of jaxtyping.
  ...
```

The same applies to `TypedDicts`, which are each checked in their own scope as
well:

```python
class Outputs(TypedDict):
  logits: Float["*b n d"]
  labels: Int["*b n 1"]
  attn: Float["*b n n"]

@kt.typechecked
def fn(net1: Outputs, net2: Outputs):  # net1 and net2 each get their own scope
  # so net1["logits"].shape might differ from net2["logits"].shape
  ...
```

### Ambiguity and Candidate Assignments

`ktyping` tracks **sets of candidate assignments** to handle ambiguities,
e.g., from `Union` types or multiple variable dimensions. For example:

```python
@kt.typechecked
def ambig(
  x: Float["b n"] | Float["b m"],
  # Here it is unclear if m=x.shape[1] or n=x.shape[1]
  y: Int["b n"],  # Now the ambiguity can be resolved (unless m=n)
  z: Int["b m"],
):
  ...
```

Ambiguity can arise from having multiple variable dimensions. E.g.:

```python
import kauldron.ktyping as kt
import numpy as np

with kt.typechecked():
  x = np.zeros((32, 7, 3))
  y = np.zeros((32,))
  kt.check_type(x, Float["*batch *data d"])
  # At this point it is unclear what *batch and *data should be. Candidates:
  # - {*batch: (), *data: (32, 7), d: 3}
  # - {*batch: (32,), *data: (7,), d: 3}
  # - {*batch: (32, 7), *data: (), d: 3}

  kt.check_type(y, Float["*batch"])
  # The ambiguity is resolved:
  # - {*batch: (32,), *data: (7,), d: 3}
```

Each check is performed against all current candidate dimension assignments.
Candidates that fail, are eliminated from the pool.
A check fails if *no* valid assignment remains.

Note: This is a key difference compared to `jaxtyping`, which uses greedy
matching for unions and doesn't support more than one variable dimension.

### Working without an active scope

Instance checks like `isinstance(x, Float["b"])` work normally even outside of
an `@kt.typechecked` function or context.
However, dimension values are not remembered or checked for consistency with
other checks.
To catch mistakes such as forgetting a `@kt.typechecked` decorator on helper
functions, `kt.check_type(...)` should be preferred over
`assert isinstance(...)`, because `kt.check_type` explicitly checks that it is
called from within a function which has an active scope (e.g. decorated with
`@kt.typechecked` or within a `with kt.typechecked()` block) and throws an
exception otherwise.

```python
def forgot_decorator(a):
  assert isinstance(a, Float["b"])  # implicitly uses "b" from the caller scope
  kt.check_type(a, Float["b"])  # throws a NoActiveScope error


with kt.typechecked():
  kt.check_type(a, Float["b"])
  forgot_decorator(a[1:])
```

Note: Both `kt.dim[...]` and `kt.shape(...)` require an active scope to work.
  This also means that they will break when deactivating typechecking.

### `kt.dim`
Provides dict-like access to the active scope:

```python
@kt.typechecked
def demo(x: Float["*b h w c"]):
  batch_size = kt.dim["*b"]  # (16, 8)
  c = kt.dim["c"]  # 3

  kt.dim["n"] = 128

  del kt.dim["h"]

  print(kt.dim)

demo(np.empty((16, 8, 32, 32, 3)))
```

which will output:

```none
dims = {
  *b: (16, 8)
   w: 32
   c: 3
   n: 128
}
```

Note: `kt.dim` only allows getting or setting of a dimension if that dim
assignment is unambiguous. If multiple conflicting assignments exist for a
dimension `kt.dim[]` will raise an `kt.AmbiguousDimensionError`.

### `kt.shape`
Resolves the spec string to a tuple of ints using the current scope:

```python
import kauldron.ktyping as kt
from kauldron.ktyping import Array, Float
import numpy as np

@kt.typechecked
def process_and_create_similar(x: Float["b n d"]) -> Array:
  return np.empty(kt.shape("b n n+1 d*2"))

y = process_and_create_similar(np.empty((4, 8, 16)))
print(y.shape) # prints (4, 8, 9, 32)
```

## Error Messages
Type checking errors raised by `ktyping` are `kt.KTypeCheckError` (a subclass
of `typeguard.TypeCheckError`) have five parts. For example:

```python
@kt.typechecked
def cross_entropy_loss(
    preds: Float["*b d"], targets: Int["*b 1"], mask: Bool["*b 1"] | None = None
) -> ScalarFloat:
  ...
```

With an error that might look like this (excluding the stacktrace):

<section  class="tabs">

## ktyping {.new-tab}

```none
[...]

KTypeCheckError: argument mask = np.bool_[32 1] has shape (32, 1) which is not
shape-compatible with '*b 1' (required by Bool['*b 1'] | None).

Origin: function 'cross_entropy_loss' at /tmp/ipython-input-3-3053638833.py:1

Arguments:
  preds: Float['*b d'] = np.f64[32 128 1000]
  targets: Int['*b 1'] = np.i64[32 128 1]
> mask: Bool['*b 1'] | None = np.bool_[32 1]

Return: ScalarFloat

Dim Assignments:
 - {*b: (32, 128), d: 1000}
```

## jaxtyping (for comparison) {.new-tab}

```none
[...]

TypeCheckError: jax.jaxlib._jax.ArrayImpl of "argument 'mask'" (jax.jaxlib._jax.ArrayImpl)
did not match any element in the union:
  jaxtyping.Bool[Array, '*b 1']: is not an instance of jaxtyping.Bool[Array, '*b 1']
  NoneType: is not an instance of NoneType

During handling of the above exception, another exception occurred:

[...]

TypeCheckError: jax.jaxlib._jax.ArrayImpl of "argument 'mask'" (jax.jaxlib._jax.ArrayImpl)
did not match any element in the union:
  jaxtyping.Bool[Array, '*b 1']: is not an instance of jaxtyping.Bool[Array, '*b 1']
  NoneType: is not an instance of NoneType

The above exception was the direct cause of the following exception:

[...]

TypeCheckError:
The problem arose whilst typechecking parameter 'mask'.
Actual value: bool[32,1](jax)
Expected type: jaxtyping.Bool[Array, '*b 1'] | None.

The above exception was the direct cause of the following exception:

[...]

TypeCheckError: Type-check error whilst checking the parameters of __main__.cross_entropy_loss.
The problem arose whilst typechecking parameter 'mask'.
Actual value: bool[32,1](jax)
Expected type: jaxtyping.Bool[Array, '*b 1'] | None.
----------------------
Called with parameters: {'preds': f32[32,128,1000](jax), 'targets': i32[32,128,1](jax),
 'mask': bool[32,1](jax)}
Parameter annotations: (preds: Float[Array, '*b d'], targets: Integer[Array, '*b 1'],
 mask: Bool[Array, '*b 1'] | None = None) -> Any.
The current values for each jaxtyping axis annotation are as follows.
d=1000
b=(32, 128)
```
</section>

Most of this message should be self-explanatory. But let's look at the
**Dim Assignments:** block in more detail:

```none
Dim Assignments:
 - {*b: (32, 128), d: 1000}
```
Since the error happened while checking the `mask`, the shapes for `*b` and `d`
have been inferred from `preds` and are thus listed here.
For cases with ambiguity, it can happen that there are multiple candidate
assignments. For example:

```python
@kt.typechecked
def foo(x: Float["b n"] | Float["b m"], y: Float["b m"]):
  ...

foo(np.zeros((8, 3)), np.zeros((7, 3)))
```

This will produce two possible assignments:

```none
[...]
Multiple Dim Assignment Candidates:
 - {b: 8, n: 3}
 - {b: 8, m: 3}
```

## Configuration
Some of the behaviour of `ktyping` can be configured either globally or on a
per-module basis. Partially disabling typechecking can be useful when
deactivating it globally would break some other code that depends on shape
inference (e.g. by using `kt.dim' or `kt.shape`).

Configuration options:

  * **`typechecking_enabled`** `[bool]`: This can be used to completely
    deactivate the typechecking via the `@typechecked` decorator and the ktyping
    specific typecheckers added to typeguard.

    Warning: This is a double edged sword, because it breaks certain
    features like `kt.dims` and `kt.get_shape` which depend on the shape
    inference done during typechecking. In that case, consider only deactivating
    it for certain modules.

  * **`jaxtyping_annotations`** `[ReportingPolicy]`: By default `ktyping` throws
    an error if it detects any `jaxtyping` annotations. This is to help with
    migration efforts, since mixing the two can lead to subtle errors.
    Use this configuration option to configure how `ktyping` deals with such
    annotations. Valid options can be found in the `ReportingPolicy` enum and
    include: `IGNORE`, `LOG_INFO`, `WARN`, `ERROR`.

The configuration can be modified in three ways:

1. Temporarily modify the config via context manager:

  ```python
  import kauldron.ktyping as kt

  with kt.Config(typechecking_enabled=False):
    ...
  ```

2. Add a module-specific override. This is done via a module-name regex. So for
example to deactivate typechecking only for `example.module` (and its
submodules):

  ```python
  import kauldron.ktyping as kt

  kt.add_config_override(
    r"example\.module",
    kt.config.Config(typechecking_enabled=False)
  )
  ```

3. Change the global configuration:

  ```python
  import kauldron.ktyping as kt

  kt.CONFIG.typechecking_enabled = False
  ```
## Differences with `jaxtyping` {#differences-with-jaxtyping}

* **Concise syntax**: `ktyping` prioritizes readability over support for static
 typecheckers:<br/>
 `kt.Float["*b h w c"]` vs `jt.Float[jt.Array, "*b h w c"]`.

* **Better error messages**: See [Error Messages](#error-messages).

* **Multiple variable dims**: Shapes like `*batch time *data` are supported by
  `ktyping` (while being an error in `jaxtyping`).
* **More expressive syntax**: Supports `+d` (one or more dims),
  `t?` (optional dim) and `3|1` (choices).
* **Non-greedy matching**: For unions and ambiguous matches, `ktyping` tracks
  all possibilities, while while `jaxtyping` uses greedy matching. For example:

  ``` python
  @jt.jaxtyped(typechecker=typeguard.typechecked)
  def foo(
    x: jt.Float[jt.Array, "a"] | jt.Float[jt.Array, "b"],
    y: jt.Float[jt.Array, "a b"]
  ): ...

  foo(f32[3], f32[7 3])  # fails because x is greedily matched with Float["a"]
  ```

* **Expressions**: Better support for expressions of unknown dimensions. E.g.
  `ktyping` correctly infers `dim` here:

  ``` python
  @kt.typechecked
  def foo(a: Float["dim+1"]) -> Float["dim"]:  # in jaxtyping this is an error
  ...
  ```

* **TypedDict namespace**: `TypedDict`s have isolated namespaces in `ktyping` so
  the same `TypedDict` annotation can be reused for multiple arguments without
  requiring identical dimensions for all items.

* **Limited `PyTree` support**: Currently, `ktyping` only supports annotations
  like `PyTree[int]`, and does not support checking named structures like
  `PyTree[int, "S"]` yet.

## Migration

### Migration from `jaxtyping`

When migrating from `jaxtyping`, apply the following changes:

* Replace `from jaxtyping` imports with `from kauldron.ktyping` imports.
* Replace `@jaxtyped(typechecker=typeguard.typechecked)` with `@kt.typechecked`
* Convert long-form `jaxtyping` annotations to `ktyping`'s short form. E.g.
  * `Float[Array, "b d"]` becomes `Float["b d"]`
* Use different `Tf...` or `X...` prefixes instead of the first argument to
  check for tensorflow or generic array classes:
  * `Int[tf.Tensor, "b d"]` becomes `TfInt["b d"]`
* False-friends / API differences:
  * `Shaped[Array, "b d"]` becomes `Array["b d"]`
  * `jt.Int` becomes `kt.SInt` (match only signed integer dtypes)
  * `jt.Integer` becomes `kt.Int` (match both signed and unsigned integer dtypes)
  * `jt.Key` becomes `kt.PRNGKey`

**Example:**

```python
# Before:
from jaxtyping import Array, Float, Int, jaxtyped  # pylint: disable=g-multiple-import, g-importing-member
import typeguard

@jaxtyped(typechecker=typeguard.typechecked)
def fn(x: Float[Array, "b d"], y: Integer[Array, "b"]) -> Float[Array, "b d"]:
    ...

# After:
import kauldron.ktyping as kt
from kauldron.ktyping import Float, Int, typechecked  # pylint: disable=g-multiple-import, g-importing-member

@kt.typechecked
def fn(x: Float["b d"], y: Int["b"]) -> Float["b d"]:
    ...
```

### Migration from `kauldron.typing`

* Replace `from kauldron.typing` imports with `from kauldron.ktyping` imports.
* False-friends / API differences:
  * `oldkt.Int` becomes `kt.SInt` (match only signed integer dtypes)
  * `oldkt.Integer` becomes `kt.Int` (match both signed and unsigned integer dtypes)
* Replace `Shape(...)` **function calls** with `kt.shape(...)`:
  In the old `kauldron.typing`, `Shape` was used both as annotation for
  `tuple[int, ...]` and as a function to construct a shape like
  `s = Shape("*b n 1")`.
  In `ktyping`, `Shape` can still be used as an annotation, but the latter
  use-case is replaced by the (lowercase) function `s = kt.shape("*b n 1")`.
* Replace `Dim()` function calls with access to `kt.dim`. E.g.
  `h = Dim("h")` becomes `h = kt.dim["h"]`
* Replace `set_shape()` with either `kt.dim` assignment or `check_type`.
  For setting a single dimension `set_shape("h", (7,))` becomes
  `kt.dim["h"] = 7`.
  For multiple dimensions `set_shape("b h w c", (1, 2, 3, 4))` can either become
  `kt.check_type(np.empty((1, 2, 3, 4)), Array["b h w c"])`, or (preffered) a
  set of single dim assignments via `kt.dim`.
* `kauldron.typing` includes a few symbols that are not part of `ktyping`, such
  as `Initializer`, `Schedule`, `Axes`, `AxisName`, `ArraySpec`, and
  `ElementSpec`. Just keep importing them from `kauldron.typing` for now.

**Example:**

```python
# Before:
from kauldron.typing import Float, Integer, Shape, typechecked  # pylint: disable=g-multiple-import, g-importing-member

@typechecked
def fn(x: Float["b d"], y: Integer["b n"]) -> Float["b d"]:
    s = Shape("b n d")
    ...

# After:
import kauldron.ktyping as kt
from kauldron.ktyping import Float, Int  # pylint: disable=g-multiple-import, g-importing-member

@kt.typechecked
def fn(x: Float["b d"], y: Int["b n"]) -> Float["b d"]:
    s = kt.shape("b n d")
    ...
```