# Changelog

<!--

Changelog follow the https://keepachangelog.com/ standard (at least the headers)

-->

## [Unreleased]

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
