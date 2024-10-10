# Subsetter

Subsetter is a Python utility that can be used for subsetting portions of
relational databases. _Subsetting_ is the action extracting a smaller set of rows
from your database that still maintain expected foreign-key relationships
between your data. This can be useful for testing against a small but
realistic dataset or for generating sample data for use in demonstrations.
This tool also supports filtering that allows you to remove/anonymize rows that
may contain sensitive data.

Similar tools include Tonic.ai's platform and [condenser](https://github.com/TonicAI/condenser).
This is meant to be a simple CLI tool that overcomes many of the difficulties in
using `condenser`.

## Installation

You can use subsetter by installing it through pip:

```sh
pip install subsetter
```

Or by using the published `msg555/subsetter` image:

```sh
docker run --rm -v "./subsetter.yaml:/tmp/subsetter.yaml" msg555/subsetter -c /tmp/subsetter.yaml subset
```

## Limitations

The subsetter tool takes an approach of "one table, one query". This means that
the subsetter will sample each table using only a single query. It cannot
support calculating a full transitive closure of foreign key relationships for
schemas that contain cycles. In general, as long as your schema contains no
foreign key cycles and no target is reachable from another target, the subsetter
will be able to automatically generate a plan that can sample your data.

# Usage

## Create a sampling plan

The first step in subsetting a database is to generate a sampling plan. A
sampling plan defines both the direct targets of the subsetter and what tables
should be brought in through indirect foreign key references.  You'll want to
create a configuration file similar to
[subsetter.example.yaml](subsetter.example.yaml), making sure to fill out the
`planner` section to tell the planner what tables you want to sample any
additional constraints that should be considered. Then you can create a plan
with the below command:

```sh
subsetter -c my-config.yaml plan > plan.yaml
```

If you inspect the generated plan YAML document you will see a syntax tree
that defines how each table will be sampled, potentially referencing other
tables. Queries can reference either source tables or previously sampled tables.
If you need to customize the way that tables are sampled beyond what the planner
can automatically produce this is the place to do it. If needed, you can even
write direct SQL here.

## Sample a database with a plan

The sample sub-command will sample rows from the source database into your
target output (either a database or as json files) using a plan generated
using the plan sub-command. By default this tool will **not** copy schema
from the source database and expects tables to already exist. If you would like
it to attempt to create tables in the output database pass the `--create` flag.
Additionally you must pass `--truncate` if you wish to clear any existing data
in the output tables that may otherwise interfere with the sampling process.

```sh
subsetter --config my-config.yaml sample --plan my-plan.yaml --create --truncate
```

The sampling process proceeds in four phases:

1. If `--create` is specified it will attempt to create any missing tables. Existing tables will not be touched even if the schema does not match what is expected.
2. If `--truncate` is specified any tables about to be sampled will be first truncated. subsetter expects there to be no existing data in the destination database unless configured to run in _merge_ mode.
3. Any sampled tables that are referenced by other tables will first be materialized into temporary tables on the source database.
4. Data is copied for each table from the source to destination.

## Plan and sample in one action

There's also a `subset` subcommand to perform the `plan` and `sample` actions
together. This will automatically feed the generated plan into the sampler,
in addition to ensuring the same source database configuration is used for
each.

```sh
subsetter -c my-config.yaml subset --create --truncate
```

# Sample Transformations

By default any sampled row is copied directly from the source database to the
destination database. However, there are several transformation steps that can
be configured at the sampling stage that can change this behavior.

## Filtering

Filters allow you to transform the columns in each sampled row using either a
set of built-in filters or through custom plugins. Built in filters allow you to
easily replace common sources of personally identifiable information with fake
data using the [faker](https://faker.readthedocs.io/en/master/) library. Filters
for name, email, phone number, address, and location, and more come built in.
See [subsetter.example.yaml](subsetter.example.yaml) for full details on what
filters exist and how to create a custom filter plugin.

## Identifier Compaction

Often tables make use of auto-incrementing integer identifiers to function as
their primary key. Sometimes we may want the identifiers in our sampled data
to be compact -- instead of retaining the value in the source database we may
want our N sampled rows to have identifiers ranging from 1 to N. This is useful
for sample data where we want to keep the identifiers easy to reference.

Any other table that has a foreign key that references one of these compacted
columns will automatically also have the column involved in that foreign key
adjusted to maintain semantic consistency.

Note that enabling compaction can have a noticable impact on performance.
Compaction both requires more tables to be materialized on the source database
and requries more joins when streaming data into the destination database.

## Merging

By default the sampler expects no data to exist in the destination database.
To get around this constraint we can turn on "merge" mode. To use merge mode all
sampled tables must be either marked as "passthrough" or have a single-column,
non-negative, integral primary key.

When enabled, the sampler will calculate the largest existing primary key
identifier for each non-passthrough table and automatically shift the primary
key of each sampled row to be larger using the equation:

```
new_id = source_id + max(0, existing_ids...) + 1
```

Passthrough tables instead will be sampled as normal except they will use the
'skip' conflict strategy which will have the effect of only inserting rows in
a passthrough table if no row with the matching primary key exists in the
destination database.

If merging multiple times it may be necessary to turn on identifier compaction
to avoid the largest identifier in each table from growing too quickly due to
large gaps.

## Multiplicity

Sampling usually means condensing a large dataset into a semantically consistent
small dataset. However, there are times that what you really want to do is
create a semantically consistent large dataset from your existing data (e.g. for
performance testing). The sampler has support for this by setting the multiplicity factor.

Multiplicity works by creating multiple copies of your sampled dataset in your
output database. To ensure these datasets do not collide it remaps all foreign
keys into a new key-space. Note that this process assumes your foreign keys are
opaque integers identifiers.

# FAQ

## How do multiple targets work

When using multiple targets each target table will be sampled entirely
independently unless another target table directly or indirectly depends on some
rows from it through a series of foreign keys. In the later case the subsetter
will sample a union of the rows from the independently sampling of the table and
those rows that other targets depend on.

## How does the subsetter use foreign keys?

The subsetter uses the foreign keys present in the database schema to understand
relationships between data and generate a sampling plan. Foreign key
relationships can be followed in both directions if need be. For example,
suppose there was a `users` and an `orders` table where `orders` had a foreign key
to the `users` table.

If `users` was sampled first the subsetter would sample `orders` from `users` by
sampling all rows from `orders` such that their corresponding user row existed.
This represents the _maximal_ set of rows that can be included without violating
foreign key constraints.

Otherwise if `orders` was sampled first the subsetter would sample `users` from
`orders` by sampling all rows from `users` such that they had at least one
`order`. This represents the _minimal_ set of rows that can be included without
violating foreign key constraints.

In general the subsetter will always sample tables in an order such that all
foreign key relationships to previously sampled tables are going in the same
direction. If they are followed in the forwards direction (as in our first case)
the subsetter will select the _intersection_ of all rows that obey each foreign
key relationship. Otherwise if they are followed in the backwards direction (as
in our second case) the subsetter will select the _union_ of all rows that obey
each foreign key relationship. This strategy ensures no foreign key
relationships are violated in the sampled data.
