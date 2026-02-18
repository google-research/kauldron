# Konfig principles

Config can be a hot topic. Why create a new config system when there are already
so many ones?

> TLDR: The goal of `konfig` is to **remove** all indirection between your code
> and config.
>
> The user just writes regular python code (following some design principles
> explained in this doc), and all their code is automatically configurable
> without **any** changes (no dependency injection or obscure magic).

## Why? (Design goals)

### Motivating example

Let's take a simple example: configuring an optimizer.

```python
optimizer = optax.adam(learning_rate=1e-5)
```

Often in config systems, to make the above configurable, users will duplicate
the params to configure in some config class, like:

```python
@dataclasses.dataclass
class AdamOptimizerConfig:
  learning_rate: float = 1e-5

  def make(self):
    return optax.adam(learning_rate=self.learning_rate)
```

This config object is then propagated down the program tree until the place it
is actually used.

We think this indirection is **boilerplate** and **unnecessary**. Very soon,
user will want to switch `adam` for a new optimizer (e.g. `adafactor`), or
customize adam `b2=` arguments. Each time users will have to update the config
and duplicate additional parameters. And with rising complexity it becomes
increasingly unclear from reading the config which optimizer will actually get
created. Config classes end-up with many unused parameters which where only
added once for a single experiment.

There must be a better way!

With konfig, there is no need to create any custom config object. The config is
written as pure Python calls. Need to customize the `b2` params? Just write it
in your config.

Rather than:

```python
cfg = AdamOptimizerConfig(learning_rate=1e-5, b2=0.99)
```

Simply write:

```python
cfg = optax.adam(learning_rate=1e-5, b2=0.99)
```

Need to switch Adam for another optimizer? Just write an arbitrarily complex one
in your config:

```python
cfg = optax.chain(
    optax.scale_by_adadelta(),
    optax.add_decayed_weights(weight_decay=0.0),
    optax.scale_by_learning_rate(learning_rate=0.003),
)
```

All python classes / functions / constants are supported without any changes to
your codebase! (`optax` doesn't know anything about konfig)

### How does it work ?

Conceptually, the idea is very simple. **A config is just a nested tree of
`dict`**. There's a **one-to-one** mapping between a nested call of Python
functions and its corresponding dict representation.

For example, the last example can be represented as:

```python
cfg = {
    '__qualname__': 'optax:chain',
    0: {'__qualname__': 'optax:scale_by_adadelta'},
    1: {'__qualname__': 'optax:add_decayed_weights', 'weight_decay': 0.0},
    2: {'__qualname__': 'optax:scale_by_learning_rate', 'learning_rate': 0.003},
}
```

Resolving the config dict back to original Python object is trivial by simply
importing the `__qualname__` symbols and calling it with the args from the
`dict`. This is done by `konfig.resolve(cfg)`.

The two codes are **identical** and return the exact same output:

<table>
  <tr>
    <td><pre><code class="language-python">optax.adam(learning_rate=1e-5)</code></pre></td>
    <td><pre><code class="language-python">konfig.resolve({
    '__qualname__': 'optax:adam',
    'learning_rate': 1e-5,
})</code></pre></td>
  </tr>
</table>

When building the config, the only magic required is to capture the module (with
`konfig.imports()`), such as calling function creates `dict`-like objects,
rather than actually executing the function.

<table>
  <thead><tr>
      <th>Without config</th>
      <th>With config</th>
  </tr></thead>
  <tr>
    <td><pre><code class="language-python">import optax

optimizer = optax.adam(learning_rate=1e-5)</code></pre></td>
    <td><pre><code class="language-python">with konfig.imports():
  import optax

cfg = optax.adam(learning_rate=1e-5)</code></pre></td>
  </tr>
  <tr>
    <td>
    Output is real Python object
    <pre><code class="language-python">isinstance(optimizer, optax.GradientTransformation)</code></pre></td>
    <td>
    Output is dict-like object
    <pre><code class="language-python">isinstance(cfg, konfig.ConfigDict)</code></pre></td>
  </tr>
  <tr>
    <td>Python code should never be mutated.</td>
    <td>Config is mutable
    <pre><code class="language-python">cfg.learning_rate = 1e-4</code></pre></td>
  </tr>
  <tr>
    <td>N/A</td>
    <td>Config is serializable
    <pre><code class="language-python">cfg.to_json() == {
    '__qualname__': 'optax:adam',
    'learning_rate': 1e-4,
}</code></pre></td>
  </tr>
</table>

> Note: Because konfig objects are simply `dict` like object, you cannot have
> control flow among them (i.e. your config can define `f(g())`, but NOT `f() +
> g()` as it's like executing `{'__qualname__': 'f'} + {'__qualname__': 'g'}`)

### Benefits

Removing the indirection between code and config makes the config code look like
regular Python code. This comes with additional advantages, like:

*   Full IDE supports out-of-the-box, including:
    *   Auto-complete
    *   Hover to show function/classes docstring
    *   Ctrl+click on a symbol jump directly to the symbol definition and usage
        (in the above case, you directly jump to `adam` implementation without
        any indirection)
*   Type checking in your config files
*   Code is copy-pastable outside of config, which is very helpful for quick
    inspection and debugging. For example, just copy-paste the
    [dataset config definition](https://github.com/google-research/kauldron/tree/main/examples/tiny_vit_imagenet.py;l=95-106;rcl=735474474)
    in Colab to create the matching Python object and inspect it.

## How ? (Best-practices)

### Rule 0: Keep config / Python code separated

Because `konfig` and standard Python code looks very similar, it's very
important to keep a clear boundary between the two.

#### Execution

*   **Before** `konfig.resolve`, all the objects inside the cfg tree are
    **only** `ConfigDict` like objects. The library will raise an error if
    adding a Python object inside the config.
*   **After** the `konfig.resolve`: all the objects are **only** real Python
    objects. There are no more `ConfigDict` objects.

#### Folder structure

*   All config code should go **inside** some `configs/` folder.
*   All the actual implementation (models, dataset implementation, additional
    logic,...) should go **outside** this `configs/` folder.

#### Colab

On Colab, it's very easy to mix regular imports with `konfig.imports()`, which
is a direct violation of Rule 0. To avoid this issue:

TLDR: **Never** use `konfig.imports()` on Colab.

*   On Colab: Instead, you can locally mock the modules with
    `konfig.mock_modules()`.

    ```python
    import optax  # NO `konfig.imports()`

    # Imports are mocked only locally
    with konfig.mock_modules():
      cfg = optax.adam(learning_rate=1e-5)  # This is a ConfigDict

    optimizer = optax.adam(learning_rate=1e-5)  # This is the real Python object
    ```

*   Outside Colab: Use `konfig.import()` for **all** imports in your config
    file. This ensures clear boundaries between configurable modules and the
    config implementation.

    ```python
    with konfig.imports():
      import optax

    cfg = optax.adam(learning_rate=1e-5)
    ```

#### Naming

When manipulating the unresolved config in your code (before `konfig.resolve`),
make it clear in the name this is a config-like object (`cfg`, `config`,
`optimizer_config`,...):

```python
# Before: object is named `cfg`
optimizer = konfig.resolve(cfg)
# After: object is named `optimizer`
```

### Rule 1: Keep complexity in Python

TLDR: Config files should be kept minimal (ideally in a single small file less
than ~300 lines) !

As your code supports more complex uses-cases, it can be tempting to add more
and more options in the config. This is a bad idea! The config files will start
to grow uncontrollably.

Instead, move the complexity to Python code, by wrapping the code in higher
level abstractions.

Note that this design rule is independent of using konfig or not. As your
program complexity grows, it's your responsibility to wrap complexity in higher
level abstractions. Remember that your Python code can directly be used without
konfig. **The config file is just the reflection of the abstractions from your
codebase**.

#### Example 1

Let's take our original example:

```python
cfg = optax.adam(learning_rate=1e-5)
```

As your use-cases get more and more complex, you start using more and more
complex optimizers.

```python
cfg = optax.chain(
    optax.scale_by_adadelta(),
    optax.add_decayed_weights(weight_decay=0.0),
    optax.scale_by_learning_rate(learning_rate=0.003),
)
```

Over time, the optimizer definition in your config starts to become too big. At
this point, it is tempting to add a `def _make_optimizer()` function in your
config file, to simplify creating the optimizer. **Resist this urge!** Factoring
out the complexity in a higher level wrapper is good, but **this abstraction
should live in the Python side. Not in the config.**

The config can then simply call your higher level abstraction:

```python
cfg = my_project.my_complex_optimizer(learning_rate=0.003)
```

#### Example 2

One common pattern where this applies is in the data pipeline. As more complex
transforms get added, the data pipeline starts to grow:

```python
cfg = kd.data.tf.Tfds(
    name="ai2dcaption",
    split="train",
    shuffle=True,
    transforms=[
        # Low-level transforms
        gm.data.Tokenize(key="prompt", tokenizer=tokenizer, add_bos=True),
        gm.data.Tokenize(key="response", tokenizer=tokenizer, add_eos=True),
        gm.data.AddNextTokenPredictionFields(...),
        kd.data.Elements(keep=["input", "target", "loss_mask"]),
        gm.data.Pad(
            key=["input", "target", "loss_mask"],
            max_length=max_length,
            truncate=True,
        ),
        kd.data.Rearrange(key=["target", "loss_mask"], pattern="... -> ... 1"),
    ],
)
```

At this point, it's a good idea to create some higher level transform in your
Python code, which can be reused across configs.

```python
cfg = kd.data.tf.Tfds(
    name="ai2dcaption",
    split="train",
    shuffle=True,
    transforms=[
        # High-level transform
        gm.data.Seq2SeqTask(in_prompt="prompt", in_response="response", ...),
    ],
)
```

#### Can I split my config into multiple files ?

Before doing this, remember rule 1. Why do you need to split config ? If it's
because the config file has grow too much, it likely indicate some of the config
logic should be moved outside of config, directly in Python.

However there's still one legitimate use-case to split the config file. If you
have some base config which defines some common fields, and the sub-configs only
customize one or two additional field.

Even then, this can be a slippery slope. Do not start to chain multiple configs.
For configs files, forking is better than inheritance. When following the above
principles, configs are small and self-contained.

### Rule 2: Keep Python code simple

#### Create a top-level abstraction for your program

Your Python code can be used as standalone, without konfig. The best way to do
this in a modular/reusable way is to define some top-level `@dataclass` which
exposes all the sub-parts user might want to customize.

For example, Kauldron experiments are a `kd.train.Trainer` class which can be
used as standalone, like:

```python
trainer = kd.train.Trainer(
    train_ds=ds,
    model=model,
    optimizer=optimizer,
    ...,
)
trainer.train()
```

Similarly, the XManager launcher is a `kxm.Experiment` dataclass which
can be called directly:

```python
xp = kxm.Experiment(
    name='My experiment',
    jobs={
        'train': kxm.Job(
            target='//path/to/my:target',
            platform='df=2x2',
        ),
    },
)
xp.launch()
```

Those top-level abstractions are very natural to be wrapped inside konfig, to
allow json serialization and CLI overwrite. In the previous example, wrapping
inside config would automatically allow user to overwrite every fields:

*   `--xp.name='My new experiment'`
*   `--xp.jobs.train.target='//path/to/my:new_target'`
*   `--xp.jobs.train.platform='df=4x4'`
*   ...

#### Python code should be immutable

In Python, mutations make it hard to track and understand side effects.

*   Config code can (and is meant to) be mutated (in the config files, on Colab,
    through CLI,...).
*   Python code should be immutable. For safety, konfig will normalize `list` ->
    `tuple`, `dict` -> `immutabledict`. If you have dataclasses, it's best to
    define them as `frozen=True`, as it is the case with the above
    `kd.train.Trainer`, `kxm.Experiment`,...

#### Do not create Config classes

The goal of konfig is to remove the indirection with your code. So do not add
the indirection back with config classes. Just directly call your `nn.Module`,
`optax.GradientTransformation`, `tfds.builder`,... in your config.
