# Changelog

<!--

Changelog follow the https://keepachangelog.com/ standard (at least the headers)

-->

## [Unreleased]

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

* Changed: removed `Checkpointer.partial_initializer` and instead added
  `cfg.init_transforms` which can be used to set multiple transformations for
  the params of the model (i.e. instances of `AbstractPartialLoader`).
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

## [0.1.0] - 2022-01-01

* Initial release

[Unreleased]: https://github.com/google-research/kauldron/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/google-research/kauldron/releases/tag/v0.1.0
