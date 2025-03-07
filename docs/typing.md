# Type Annotations and Shape checking

## Shape Annotations

Kauldron has support for shape annotations (powered by
[jaxtyping](https://github.com/google/jaxtyping) and
[typeguard](https://github.com/agronholm/typeguard)) that can be automatically
runtime-checked. For example:

``` python
from kauldron.typing import typechecked, check_type, Float, Int, Shape, Bool

@typechecked
def masked_gray_image(
    img: Float["*b h w c"],
    mask: Bool["*b h w #c"]
) -> Float["*b h w 1"]:
    return jnp.mean(img*mask, axis=-1, keepdims=True)

# this works
img = masked_gray_image(
    jnp.zeros((8, 64, 64, 3)),
    jnp.zeros((8, 64, 64, 1), dtype=jnp.bool_))
# this will raise an TypeCheckError complaining about mask not being
# an instance of Bool["*b h w #c"]
img = masked_gray_image(
  jnp.zeros((8, 64, 64, 3)),
  jnp.zeros((8, 32, 32, 1), dtype=jnp.bool_))

```

Note that shape specifications are checked for consistency in the scope of the
`@typechecked` decorator (i.e. the function), thus the example above fails
because the `h w` of mask doesn’t match the `h w` of img.

### Shape Spec Syntax
The shape-spec should be a string of space-separated dimension specs, such as
`"a b c d"`. Each dim can be either an:

  * int: fixed-size axis, e.g. `"28 28"`.
  * str: variable-size axis, e.g. `"h"`, `"batch"`, or `"channels"`.
  * Prepend `*` to a dimension to indicate that it can match multiple axes,
    e.g. `"*batch c h w"` will match zero or more batch axes.
  * Prepend `#` to a dimension to indicate that it can be of that size OR equal
    to one -- i.e. broadcasting is acceptable. This can be combined with `*` to
    denote a variable number of broadcastable dimensions (e.g. `*#b`).
  * Prepend `_` to a dimension to disable any runtime checking of that dimension
    (so that it can be used just as documentation). This can also be used as
    just `_` on its own: e.g. `"b c _ _"`.
  * Ellipsis (`...`) corresponds to an anonymous zero or more axes
    (equivalent to `*_`).
  * A symbolic expression (**without spaces!**) in terms of other variable-size
    axes, e.g.:

    ```python
    def remove_last(x: Float[Array, "dim"]) -> Float[Array, "dim-1"]:
    ```
    NOTE: Symbolic expressions only work for axis for which the size is known
    dimensions at the time they are checked. So the following will raise an
    exception
    `remove_last(x: Float[Array, "dim+1"]) -> Float[Array, "dim"]`
    because at the time of calling `remove_last`, the typechecker doesn't know
    the value for `dim` and cannot evaluate `dim+1`.

    Supported symbolic operations include:
    * binary operations `+ - * / // % **`
    * parenthesis e.g. `2*(h+1)`
    * functions `min`, `max`, `sum`, and `prod`
      (e.g. `min(h,w)` or `prod(*batch)`)
  * f-string: Expressions in curly braces are evaluated as if they were part of
    an f-string with the namespace containing the arguments of the function.
    For example:

    ``` python
    def __call__(self, tokens: Float["*b {num} {self.hidden_dim}"], num: int):
    ```

### Manual Shape Checking

Shape specs can also be manually checked using check_type, which also checks
dimensions for consistency within the current scope (only if inside a function
 decorated with `@typechecked`).

``` python
@typechecked
def manual_check(img: Float["*b h w c"], mask):
    check_type(mask, Bool["*b h w #c"])  # also checks for consistency with img
    return jnp.mean(img*mask, axis=-1)
```

Using `assert isinstance(mask, Bool["*b h w #c"])` would also work but
produces less informative error messages.

NOTE: Within the scope of a `@typechecked` function, both `check_type` and
`isinstance` checks have side-effects if they are successful: Any matched
dimensions are remembered, and used in all future checks. For example
(inside of a `@typechecked` function!) either of these two checks might pass but
not both together:
`check_type(np.eye(4), Float["a a"])`, `check_type(np.eye(5), Float["a a"])`

### Complex Type Annotations
Both `@typechecked` and `check_type` support other (non-array) types as well as
compound type annotations and via the usual syntax. For example:

  * Optional `Optional[Float["b n d"]]`
  * Union `Float["b h w 3"] | Uint8["b h w 3"]`
  * tuple, list, dict etc: `dict[str, Float["..."]]`
  * built-in types `Float[""] | float | int`
  * Dataclasses: e.g.

  ``` python
  @dataclasses.dataclass
  class Batch:
    image: UInt8['b h w c']
    label: Int['b']


  @typechecked
  def get_mock_data() -> Batch:
    return Batch(image= np.zeros((8, 256, 256, 3), dtype=np.uint8),
                label=np.zeros((7,), dtype=np.int32))


  get_mock_data()
  # TypeCheckError: value of key 'label' of the return value was i32[7]
  # which is not shape-compatible with 'b'
  ```
  * and many more like `TypedDict`, callables, protocols, literals, etc.

### Accessing Dimensions and Shape Construction
Inside of a `@typechecked` function the stored shape information can be accessed
and used via `kd.typing.Dim` and `kd.typing.Shape` functions. This allows very
readable construction of arrays etc.. E.g.

``` python
@typechecked
def create_mask(img: Float["*b h w c"]) -> Bool["*b h w 1"]:
  out_shape = Shape("*b h w 1")  # returns a tuple of ints
  c = Dim("c")  # returns an int
  # e = Dim("e") # error: unknown dimension e
  return np.zeros(out_shape, dtype=np.bool_)
```

### Conventions and Best Practices

All models and functions should ideally support arbitrary batch shapes, i.e.
<span style="background:#0f03">`"*b‌ h‌ w‌ c"`</span> rather than
<span style="background:#f003">`"b h w c"`</span>.
Note that this implies indexing should always be used from the right, i.e. in
general, only negative indexing should be used, even if positive indexing would
be an option.
For example, for an image batch with shape `"b h w c"`, use
<span style="background:#0f03">`images.shape[-2 ]`</span>
instead of <span style="background:#f003">`images.shape[3]`</span> to refer to
the width dimension.
Furthermore, scalar arrays (e.g, grayscale images, labels, …) should generally
have shapes that are right-padded with 1s to allow implicit broadcasting.
For example: <span style="background:#0f03">`logits: Float["*b n"]` and
`labels: Int["*b 1"]`</span> rather than <span style="background:#f003">
`logits: Float["*b n"]` and `labels: Int["*b"]`</span>.
This choice allows all implementations to handle arbitrary batch dimensions out
of the box without the need for explicit reshapes or vmaps. It also allows many
operations to generalize e.g. from images to video, or from single-view to
 multi-view.

Regarding naming of axis, prefer using lower-case everywhere
(this is more consistent with XLA and other annotations):

  * Batch dims `*b`
  * Sequence length `t` (for use in sequential data such as video, audio, and text)
  * Number of tokens `n` (for non-sequential data)
  * Image height width and channels: `h w c`
  * Features `d`
