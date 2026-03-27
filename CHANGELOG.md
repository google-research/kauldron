# Changelog

<!--

Changelog follow the https://keepachangelog.com/ standard (at least the headers)

-->

## [Unreleased]

## [1.4.1] - 2026-03-27

* `kd.ktyping`:
  * [Fix] Fix `PyTree[T]` traversing into registered types (e.g. `flax.struct.dataclass`).
  * [Fix] Fix `as_np_dtype` lookup when `torchapix` is imported.
  * [Fix] Fix `UNKNOWN_DIM` formatting in error messages.

* `kd.konfig`:
  * [Fix] Add hint to import errors if they look like the internal repo prefix is missing.
  * [Fix] Update Error message propagation in `module_configdict`.

* `kd.random`:
  * [Extended] Support `kd.random.PRNGKey(seed)`.

* `kd.cli`:
  * [New] Add `run eval_shape` command.
  * [Changed] Refactor/beautify CLI help aesthetics.

* `kd.contrib`:
  * [New] Add a configurable library for `kd.contrib.data.SSTable`.

## [1.4.0] - 2026-03-11

Highlights:

* new `kd.cli` tool
* `py::` flag syntax for konfig
* expanded `kd.ktyping` with `@typechecked` context managers / dataclass /
generator support
* meta-configs with `@dataclasses.dataclass`
* new Nnx wrapper API,
* many quality-of-life improvements across data, evals, metrics, and checkpointing.

* `kd.cli`:
  * [New] `kd.cli`: New kauldron CLI tool — a `:trainer_cli` binary
    automatically created by `kauldron_binary`, using noun-verb command style
    (e.g. `kd data element_spec`), with each command mirrored as a Python
    function.

* `kd.konfig`:
  * [New] `py::` flag value parsing — specify Python objects directly from CLI
    flags (e.g. `--cfg.xxx="py::my_module.MyObject(x=1)"`), with Lark grammar
    and alias resolution.
  * [New] Meta-configs: `@dataclasses.dataclass`-based config declaration with
    `__args__` CLI overrides and lazy config building.
  * [New] `konfig.export()`: serialize Python objects (dataclasses, arrays,
    dicts) to a dict representation.
  * [New] `unfreeze()` function for unfreezing `ImmutableDict`s.
  * [Extended] Support (named) tuples as dictionary keys in serialization.
  * [Extended] `DEFINE_config_file` accepts a `required` argument.
  * [Extended] `konfig.resolve` highlights where the original `ConfigDict` was
    created in tracebacks.
  * [Extended] Better error messages for config resolution failures and
    `FieldReference` with path tracking.
  * [Extended] Allow `konfig.restricted()` without specifying type.
  * [Changed] Always use literal evals when parsing flags — `--cfg.xxx=None` is
    now `None` rather than `'None'`.
  * [Changed] Two-stage resolution is now the default for `DEFINE_config_file`.
  * [Changed] Deprecated konfig property now raises an error.
  * [Fix] Fix JSON args parsing, list arguments from CLI, `unfreeze` bug,
    dynamic resolve trigger, `temporary_imports()` thread-safety, `BaseConfig`
    hash, and resolve errors.

* `kd.ktyping`:
  * [New] `ArraySpec`, `ElementSpec`, `PRNGKeyLike` types.
  * [New] `kt.isinstance` function for bool-returning type checks.
  * [New] Basic `PyTree[T]` annotation with runtime checking and path-aware
    errors.
  * [New] PyTree structure specs.
  * [New] Per-module config system.
  * [New] Warnings when mixing ktyping and jaxtyping.
  * [Extended] `@typechecked` now supports: context managers (`with
    typechecked():`), nested context managers, dataclasses, generator functions,
    methods / class methods / static methods.
  * [Extended] Improved shape inference for binary operations (e.g.
    `Array["a+1"]`).
  * [Extended] TensorFlow and XArray type support.
  * [Breaking] Rename `get_shape()` → `shape()` and `kt.dims` → `kt.dim`.
  * [Fix] Fix shape checking with TF Tensors, PRNGKey dtype for new-style JAX
    keys, Scalar type checking, array type union checking, broadcastable dims,
    typeguard 4.5.0 compatibility.

* `kd.nn`:
  * [Changed] New Nnx wrapper API — natively compatible with kontext keys,
    supports catching intermediates.
  * [New] Nnx wrapper documentation.

* `kd.data`:
  * [New] `LazyBagDataSource` for lazy loading of bag data.
  * [New] `SelectFromDatasets` for dataset mixtures with user-defined selection.
  * [New] `shard_by_process` to control dataset sharding behavior.
  * [New] `AddBias` transform.
  * [New] Random transforms for PyGrain pipelines.
  * [New] Padding batches with `batch_drop_remainder='pad'`.
  * [Extended] `CenterCrop` supports nD arrays.
  * [Extended] `Resize` supports min/max size targets.
  * [Extended] `RepeatFrames` works with both TF and NumPy/JAX arrays.
  * [Changed] Default `Resize` method for float inputs is now `bilinear` for
    JAX/NumPy (remains `area` for TF).
  * [Fix] Fix `ElementWiseRandomTransform`, `grain.shuffle` seed range,
    unknown-length datasets, `Tfds.decoders` with `ImmutableDict`, `Resize`
    device transfer, `element_spec` global vs device-local, filter transform
    type checking, walrus operator breaking TF autograph.

* `kd.kontext`:
  * [Extended] `set_by_path` returns the list of concrete modified paths when
    using glob patterns (`**`, `*`).
  * [Fix] Fix `kontext.imports()` errors in docs and `CONFIG_IMPORT`
    placeholder for Colab.

* `kd.train`:
  * [New] `NoOpTrainStep` for use cases skipping training.
  * [New] `checkify` support on `TrainStep.init` and `Evaluator.evaluate`.
  * [New] Expose `KDMetricWriter`, `Orchestrator`, `DirectoryBuilder` as public
    APIs for subclassing.
  * [New] `konfig_freeze` option to skip immutabledict conversion.
  * [Extended] `MultiTrainStep` subupdates for better logging.
  * [Extended] Device-to-host transfer for checkify error checking.
  * [Breaking] Rename `ShardingStrategy.ds` → `ShardingStrategy.batch`.
  * [Removed] Deprecate `CollectingState`.
  * [Fix] Fix sweeps bug with default `config_args`, `partial_updates` with
    integer keys, `transfer_guard` with `jax_debug_nans`, `MultiTrainStep`
    hashability, `ml_python+adhoc` error, `FSDPSharding` type annotation.

* `kd.evals`:
  * [New] `CheckpointedEvaluator` for resumable evaluations.
  * [New] Skip initial step 0 option.
  * [New] `NoopExporter`.
  * [New] `eval_step` added to `Evaluator`.
  * [Extended] Allow skipping checkpointing in `TrainEvaluator`.
  * [Changed] `NoOpCheckpointer` is default for `SamplingEvaluator`.
  * [Changed] `_ConcatContainer` speeds up `concat_field` aggregation.
  * [Fix] Fix non unresponsive with custom dataset in eval, duplicated
    `job_group` in eval_only.

* `kd.metrics`:
  * [New] `finalize` method for metric states.
  * [Extended] Support predicted labels (not just logits) in `Accuracy`.
  * [Extended] `min_field`, `max_field` for `AutoState`.
  * [Extended] Pytree support for `auto_state.sum_field`.
  * [Extended] Better error reporting for merging / finalizing / computing.
  * [Fix] Fix one-hot class count in segmentation metrics, `finalize()` bugs,
    `CollectingState.merge` performance.

* `kd.summaries` / `kd.vizual`:
  * [New] Confusion matrix summary.
  * [Extended] `ShowSegmentations`: `palette`, `edge`, and `hard` options.
  * [Extended] `ShowImages`: `cmap` option.
  * [Extended] `ImageGrid` convenience method.
  * [Fix] Fix `ShowDifferenceImages` type-check and JAX/numpy mismatch,
    `ShowImages` RGB output with NaN values, `bfloat16` support, integer arrays
    in `ShowSegmentations`.

* `kd.optim`:
  * [New] `ema_weights` wrapper for EMA weight tracking.
  * [Fix] Fix debias logic in `ema_params`.

* `kd.ckpts`:
  * [New] Custom Orbax preservation policy support.
  * [Removed] Remove deprecated `AbstractPartialLoader` alias.
  * [Fix] Fix EMA params loading for frozen params, checkpoint loading, snapshot
    directory race conditions, named tuple compatibility in parameter paths.

* `kd.contrib`:
  * [New] `NpzWriter`: metric writer saving array summaries to `.npz` files.
  * [New] `TreeUnflattenForKey` PyGrain transform.
  * [New] `GifVideoWriter` and `ShowVideosAsGif` for GIF video summaries.
  * [New] NNX-to-Linen wrapper `linen_from_nnx()`.
  * [New] Model exporter for JAX export.
  * [New] Online Mean+Covariance estimation state, `merge_field` in auto-state.
  * [Extended] `concat_field` works with pytrees.

* `kd.contrib.millstone`:
  * [New] New doc.
  * [Extended] Custom Borg runtime, eval dataset support, troubleshooting guide.
  * [Removed] Delete deprecated Millstone API.
  * [Fix] Fix Pathways server termination during eval.

* `kd.xm`:
  * [New] `jax_log_compiles` configurable via `xp.debug.jax_log_compiles`.
  * [Extended] Launch configargs support.
  * [Fix] Fix `cuda_compress` flag for non-GPU builds, duplicated `job_group`.

* `kd.random`:
  * [Changed] Move truncation of `as_seed()` to uint32 inside `as_seed()`.

## [1.3.0] - 2025-07-15

* Various bug fixes and improvements

## [1.2.2] - 2025-04-28

* `--xp.debug.catch_post_mortem` flag now works externally as well
* Fixed a problem with `init_transforms` that affected `optim.decay_to_init`
* Further removed deprecated summaries protocol
* Fixes regarding `grain` workers and non-thread-safe imports such as `einops`
* Lifted jit restriction for merging `auto_state.sum_field`
* Added support for `ImmutableDict` inside `konfig`
* Added `ShowTexts` summary
* Make grain an optional dependency on Windows
* Reduced logging noise
* Several other minor bugfixes

## [1.2.1] - 2025-03-10

*   Minor bug fixes.

## [1.1.2] - 2025-02-11

*   Fix `kd.sharding.FSDPSharding()` to supports `jax.ShapeDtypeStruct`
*   `kd.data`:
    *   `kd.data.py.PyGrainPipeline` supports direct indexing (`ds[0]`).
    *   `kd.data.py.HuggingFace` supports
*   Typeguard / typechecking
*   +various changes and improvements

## [1.1.1] - 2025-02-11

*   Restore numpy 1.26 compatibility

## [1.1.0] - 2025-02-07

*   Add `kd.nn.WrapperModule` to make a inner-module transparent with
    respect of Flax modules.
*   Many other changes...

## [1.0.0] - 2024-11-21

* `kd.kontext.Path` now supports tensor slicing. So for example using keys like
  `"interm.tensor[..., 0:10, :, -1]"` will now work as expected.
* `kd.nn.interm_property` now supports accessing any intermediates from within
  the model via `self.interm.get_by_path('path.to.any.module.__call__[0]')`.
* Deprecated: Remove `--xp.sweep_info.names=` flag. Instead, sweep are unified
  under `--xp.sweep` (see: https://kauldron.rtfd.io/en/latest/intro.html#sweeps)
* Add `kd.data.loader.TFData` for arbitrary `tf.data` pipelines
* Add `kd.data.InMemoryPipeline` for small datasets that fit in memory
* Add `kd.knn.convert` to convert any Flax module to klinen.
* Add `kontext.path_builder_from` to dynamically generate keys for the config
  with auto-complete and static type checking.
* Add `kd.data.BatchSize(XX)` util
* Breaking: `Evaluator(run_every=XX)` kwarg is removed. To migrate, use
  `Evaluator(run=kd.evals.RunEvery(XX))`
* Added: Eval can now be launched in separate job:

  ```python
  cfg.evals = {
      'eval_train': kd.evals.Evaluator(
          run=kd.evals.RunEvery(100),  # Run along `train`
      ),
      'eval_eval': kd.evals.Evaluator(
          run=kd.evals.RunXM(),  # Run in a separate `eval` job.
      ),
  }
  ```

* New XManager launcher

  ```sh
  xmanager launch third_party/py/kauldron/xm/launch.py -- \
        --cfg=third_party/py/kauldron/examples/mnist_autoencoder.py \
        --cfg.train_ds.batch_size=32 \
        --xp.sweep \
        --xp.platform=a100 \
        --xp.debug.catch_post_mortem
  ```

  This unlock many new features:

  * Based on `konfig` (so everything can be deeply configured).
  * Customize the work-unit directory name, default to
    `{xid}/{wid}-{sweep_kwargs}`, for better TensorBoard
    work-unit names.
  * Sweep on XManager architecture:

    ```python
    def sweep():
      for platform in ['a100', 'v100']:
        yield {'cfg.xm_job': kxm.Job(platform=platform)}
    ```

  * Possibility to launch eval jobs in a separate job
  * `ml_python` & xreload support for much faster XM iteration cycles
  * New `kd-xm` colab to quickly launch experiments without even having to open
    a terminal

* Changed: removed `Checkpointer.partial_initializer` and instead added
  `cfg.init_transform` which can be used to set multiple transformations for
  the params of the model (i.e. instances of `InitTransform`).
* Changed: `konfig.imports()` are not lazy by default anymore (config don't
  need to be resolved in `with ecolab.adhoc()` anymore!)
* Added:
  * `kd.optim`: Optimizer / optax utils
  * `kd.eval`: Eval moved to their separate namespace
* Changed: Resolved konfig can now use attribute access for dict:
  * Before (still supported): `cfg.train_losses['my_loss']`
  * After: `cfg.train_losses.my_loss`
* Added: `kd.nn.set_train_property` to change the `self.is_training` property
  value inside a model:

  ```python
  class MyModule(nn.Module):

    @nn.compact
    def __call__(self, x):
      with kd.nn.set_train_property(False):
        x = self.pretrained_encoder(x)
  ```
* Added: `kd.nn.ExternalModule(flax_module)` to use any external flax modules
  inside Kauldron.
* **And many, many more changes...**

## [0.1.0] - 2022-01-01

* Initial release

<!-- mdlint off(LINK_UNUSED_ID) -->

[Unreleased]: https://github.com/google-research/kauldron/compare/v1.4.0...HEAD
[1.4.0]: https://github.com/google-research/kauldron/releases/tag/v1.3.0...v1.4.0
[1.3.0]: https://github.com/google-research/kauldron/releases/tag/v1.2.2...v1.3.0
[1.2.2]: https://github.com/google-research/kauldron/releases/tag/v1.2.1...v1.2.2
[1.2.1]: https://github.com/google-research/kauldron/releases/tag/v1.2.0...v1.2.1
[1.2.0]: https://github.com/google-research/kauldron/releases/tag/v1.1.1...v1.2.0
[1.1.1]: https://github.com/google-research/kauldron/releases/tag/v1.1.0...v1.1.1
[1.1.0]: https://github.com/google-research/kauldron/releases/tag/v1.0.0...v1.1.0
[1.0.0]: https://github.com/google-research/kauldron/releases/tag/v0.1.0...v1.0.0
[0.1.0]: https://github.com/google-research/kauldron/releases/tag/v0.1.0
