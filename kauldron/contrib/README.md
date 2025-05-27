# Contrib

Contrib folder is a collection of Kauldron components (datasets, models,
metrics,...) that are reusable across multiple projects.

This facilitates discoverability and project sharing.

Contributions are welcome!

## Contrib and extra deps

For discoverability/accessibility, all contrib are exposed under a single
namespace: `kd.contrib` (no extra import required).

However, at the same time, users should not have to pay the cost of additional
dependencies for features they don't use.

For this reason, using some contrib features may require additional deps.
