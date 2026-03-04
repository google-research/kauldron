# Changelog

<!--

Changelog follow the https://keepachangelog.com/ standard (at least the headers)

-->

## [Unreleased]

## [1.4.0] - 2026-03-04

### Added

* `kd.ktyping`: Major new runtime typechecking tool.
* New `kauldron` CLI tool.
* New Model exporter (`kd.evals.ModelExporter`).
* Added support for setting Python objects from CLI using `py::` prefix.
* Added NNX wrapper module and documentation
* `kd.optim.ema_weights` for exponential moving average of parameters.
* `konfig.export()` function for serialization of any object as json.
*  Added support for parameterizing `get_config` with `ConfigArgs`.
* `kd.metrics`: `finalize` method for metric states, `merge_field`,
  `min_field`, `max_field`, `concat_field` pytree support, online
  Mean+Covariance estimation summary.
* Added `kd.data.py.CenterCrop`, `kd.data.py.ElementWiseRandomTransform`,
  random transforms for PyGrain pipelines.
* `kd.data.py.SelectFromDatasets` for data mixtures.
* Support padding batches with zeros in PyGrain pipelines.
* `NoOpTrainStep` and `NoOpCheckpointer`.
* `ShowConfusionMatrix` and `AddBias` transform.
* `ShowSegmentations` supports predicted labels and plotting without logits.
* `cmap` option for `ShowImages`.
* `kd.data.Resize` generalized to min/max size.
* `kd.evals.CheckpointedEvaluator` to allow resumption of evaluations.
* `eval_step` added to `Evaluator`.
* Option to skip initial step 0 for evaluators.
* `konfig_freeze` option for configs.
* Custom orbax preservation policies.
* `xp.debug.jax_log_compiles` flag.

### Changed

* **Breaking:** Removed deprecated `AbstractPartialLoader` alias.
* `ShardingStrategy.ds` renamed to `ShardingStrategy.data`.
* `CollectingState` deprecated in favor of `auto_state` fields.

### Fixed

* `ema_params` debias logic fix.
* `kd.random.as_seed` truncation fix.
* Various checkpoint loading fixes.
* Fix hang when using custom dataset in eval.
* Fix `ml_python` + adhoc error.
* Fix `konfig` resolve errors and better traceback for Config objects.
* Fix json args parsing.
* Fix `kd.data.py.Tfds.decoders` with `ImmutableDict`.
* Fix `checkify` handling for static inspect.
* Compatibility fix for typeguard 4.5.0.
* Various `ShowSegmentations` and image summary fixes.
* Fix `transfer_guard` with certain JAX versions.
* Fix duplicated `job_group` in `eval_only`.
* Improved `FieldReference` error messages in `konfig`.


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
[1.2.1]: https://github.com/google-research/kauldron/releases/tag/v1.1.2...v1.2.1
[1.1.2]: https://github.com/google-research/kauldron/releases/tag/v1.1.1...v1.1.2
[1.1.1]: https://github.com/google-research/kauldron/releases/tag/v1.1.0...v1.1.1
[1.1.0]: https://github.com/google-research/kauldron/releases/tag/v1.0.0...v1.1.0
[1.0.0]: https://github.com/google-research/kauldron/releases/tag/v0.1.0...v1.0.0
[0.1.0]: https://github.com/google-research/kauldron/releases/tag/v0.1.0
