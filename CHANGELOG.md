# v0.4.0

- Unified config file
- Added ssl configuration support to source/output database connections
- Fixed bug in infer foreign key logic that prevented it from detecting
  duplicate primary keys
- Added option to infer foreign keys across all schemas or just within schemas.
- Added database output `conflict_strategy` option
- Added `include_depencencies` planner option
- Fix bug generating plans with a target column in constraint with multiple options

# v0.3.1

- Update documentation
- Improve error messaging
- Update output formatting
- Add support for creating output schema if it does not exist
- Removed `normalize_foreign_keys` option that is no longer needed
- Like constraints in target constraints are now 'and'ed together as with all
  other constraints instead of 'or'ed
- Changed optional dependency names to match sqlalchemy
- Added sqlite support (mostly to help with tests)

# v0.3.0

- Added support for remapping schema and table names in sampling phase
- Patch support for JSON data types
- Improve live database testing
- Replaced a lot of custom dialect handling with sqlalchemy core api
- Plan output no longer outputs if tables should be materialized. Sampler now
  calculates if it needs to materialize a table itself.
- Fixed significant bug in solver that prevented finding a valid plan in many
  scenarios. Now planning will only fail if there is a forward cycle of foreign
  keys.
- Added support in mysql dialects for sampling across multiple foreign keys to
  the same table. Previously this would result in an error due to attempting to
  reopen a temporary table which is not supported in mysql.
- Ordering of tables in the produced plan no longer matters

# v0.2.0

Added postgres support alongside the existing mysql support. To install needed
dialect pacakages install either `subsetter[mysql]` or `subsetter[postgres]`
depending on your needs. Supports subsetting across dialects as well.

Added new plan config option `normalize_foreign_keys` disabled by default. The
old behavior was equivalant to this being enabled but I felt this behavior may
be a bit surprising to be enabled by default.  See documentation in
[planner_config.example.yaml](planner_config.example.yaml) for what this flag
does.

Plan output format has significantly changed. Instead of outputting SQL it now
outputs a syntax tree that then can be formed back into appropriate SQL
depending on the destination dialect. You can use the `--sql` flag to have the
planner output a plan that includes raw SQL instead to get back to something
closer to the old behavior. This will break cross-dialect sampling, however.
