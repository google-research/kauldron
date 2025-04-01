# KTyping

TLDR: ktyping is a library for annotating and run-time checking of array types
including dtypes and shapes. It is an alternative to `jaxtyping`, and fixes some
 of its problems.



## Shape Spec Mini-Language

ShapeSpec: DimExpr separated by spaces
DimExpr: ( ) + - * / // % ** - min max sum prod
Dim: [`*`|`+`][`#`]NAME[`?`]

Dimension:
  * length modifier: `*` or `+` (optional)
  * broadcast `#` (optional) (`#a` is equivalent to `a|1`, but `*#b` is not quite the same as `*b|*1`)
  * NAME python identifier (anonymous if it starts with a `_`)
  * `...` shorthand for `*_`
  * `?` suffix is used to denote an optional dimension

## Problems with jaxtyping

### Verbose Annotations (Done)
To also support static typechecks via pytype, jaxtyping has opted to use the
`Float[ArrayT, "*b h w c"]` syntax.
This allows pretending that `Float` is just an alias for `Annotated` and let's
pytype check the argument against `ArrayT`.

While this is a nice feature, it is IMHO not worth the verbosity. I prefer
`Float["*b h w c"]` because it is much more readable. And the value of static
typechecking is not very high here because we are already checking these types
at runtime.

### Greedy matching with sideeffects (Done)
All successful typechecks have the side-effect of setting dimensions.
That means for Unions in which multiple matches are possible the first one
is greedily chosen and determines the dimensions for all following arguments.

``` python
@typechecked
def foo(x: Float["a"] | Float["b"], y: Float["a b"])
  ...

foo(f32[3], f32[7 3])  # fails because x is greedily matched with Float["a"]
```

Ideally the typechecking would go through all elements of the Union and remember
all possible dimension assignments.
Future checks would weed out these possibilities, and only fail if none remain.

### Expressions fail for unknown dimensions
``` python
# this fails
@typechecked
def foo(a: Float["dim+1"]) -> Float["dim"]:
  ...

# this succeeds
@typechecked
def foo(a: Float["dim"]) -> Float["dim-1"]:
  ...
```

For simple arithmetics like this example dims could easily be inferred and
should be.
Solving this in general requires solving systems of equations and likely not
Consider checking if `f32[25, 4, 28]` matches `Float["a*b+1 a-c (b+c)*c]`.

Important question is what the behaviour should be if the dimension cannot be
inferred. Currently jaxtyping throws an error immediately.
But it might be better to remember unresolved constraints until the end of the
function, and check them again after intermediate typechecks had the chance to
populate more of the unknown dims.

### Dataclass inputs are not checked
``` python
@dataclasses.dataclass
class Bar:
  a: Float["b"]
  b: Float["b"]

@typechecked
def foo(f: Bar):
  pass

foo(Bar(a=np.zeros((3,)),
        b=np.zeros((8,))))  # does not fail
```

It is possible to decorate the dataclass instead, which would check the fields
at `__init__` time. But it might be undesirable or impossible to add this to the
dataclass. And in fact it does also seem to not work with the new-style
decorator.


### TypedDicts are not Namespaced

TypedDicts are useful for defining a format. But when reusing them
within the same function they share namespaces. So in the following example
`x`, `y` and the output of the function would have to have the same shapes:

``` python
class EncoderOutput(typing.TypedDict):
  tokens: Float["*b n d"]
  masks: Bool["*b n 1"]

@typechecked
def concat_outputs(x: EncoderOutput, y: EncoderOutput) -> EncoderOutput:
  ...
```

Currently the only (ugly) way around this is to define multiple different dicts
(e.g. `EncoderOutput1`, `EncoderOutput2`, ...).

TypeDicts should come with their own namespace, and only be checked for
consistency inside.

### Incompatible with typeguard >= 3  (obsolete)
Their combination leads to all kinds of nasty problems.
Many of which I have now worked around, but it is all very brittle and hacky.


## Feature requests



### Optional dimensions (Done)

`Float["b t? n c"]`.


### Multiple variable dimensions (Done)

`Float["*b n h *rest"]`

### Dtype Union (done)

`(Float|Int)["b h w"]`

### Or for dimensions (Done)

`Float["*b d|c"]`


### Broadcastable Variable Dim (Done)
What is the intended behavior of `x: Float["*#B"]` and `y: Float["*B"]`?

Assume we encounter `x = f32[8 1]`
Does that mean `y = f32[8 8]` is valid? -> Yes

First y then x? Should also be valid! Order shouldn't matter!

So that means we need to encode the shape after x as (8, unknown).

what if we encounter f32[8 1] and f32[1 8] both as *#B ?
I would assume that this leads to *B being (8, 8).

### Name collisions (Done)
Is this a valid shape?
`x: Float["*B B"]`
I don't think it should be.

Same for `x: Float["*B"], y: Float["B"]`. This should throw an error.
Maybe unless B is actually a single dim... hmmm





## Disabling Typechecks (TODO)

  * sometimes it can be helpful to disable typechecking (esp. for debugging)
  * Problems arise when this breaks checks like assert_caller_has_active_scope
    and features like `dims.a` or `dims.shape("*b c+1")`.
  * Ideally we would support:
    - fine-grained control over when / where to disable typechecks
    - allow disabling only some checks (e.g. only dtype, only non-array-checks, ...)
    - a fallback "sloppy" or "best-effort" mode for shape inference even if
      disabling checks. That mode could issue warnings rather than raising
      errors, ignore ambiguities, and skip some of the stricter checks.



