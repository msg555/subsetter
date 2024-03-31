
# v0.2.0

Added postgres support alongside the existing mysql support. To install needed
dialect pacakages install either `subsetter[mysql]` or `subsetter[postgres]`
depending on your needs. Supports subsetting across dialects as well.

Added new plan config option `normalize_foreign_keys` disabled by default. The
old behavior was equivalant to this being enabled but I felt this behavior may
be a bit surprising to be enabled by default.  See documentation in
[planner_config.sample.yaml](planner_config.sample.yaml) for what this flag
does.

Plan output format has significantly changed. Instead of outputting SQL it now
outputs a syntax tree that then can be formed back into appropriate SQL
depending on the destination dialect. You can use the `--sql` flag to have the
planner output a plan that includes raw SQL instead to get back to something
closer to the old behavior. This will break cross-dialect sampling, however.
