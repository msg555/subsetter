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
using `condenser.

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
sampling plan defines the queries that will be used to sample each table.
You'll want to create a configuration file similar to
[planner_config.example.yaml](planner_config.example.yaml) that tells the
planner what tables you want to sample along with any additional constraints
that should be considered. Then you can create a plan with the below command:

```sh
subsetter plan -c my-config.yaml > plan.yaml
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
using the plan sub-command. This tool will **not** copy schema from the source
database. Any sampled tables must already exist in the destination database.
Additionally you must pass `--truncate` if you wish to clear any existing data
in the sampled tables that may interfere with the sampling process.

```sh
subsetter sample --config my-sample-config.yaml --plan my-plan.yaml --truncate
```

The sampling process proceeds in three phases:

1. If `--truncate` is specified any tables about to be sampled will be first truncated.
2. Any sampled tables that are referenced by other tables will first be
materialized into temporary tables on the source database.
3. Data is copied for each table from the source to destination.

The sampler also supports filters which allow you to transform and anonymize your
data using simple column filters. See
[sampler_config.sample.yaml](sampler_config.sample.yaml) for more details on what
filters are available and how to configure them.

## Plan and sample in one action

There's also a `subset` subcommand to perform the `plan` and `sample` actions
together. This will automatically feed the generated plan into the sampler,
in addition to ensuring the same source database configuration is used for
each.

```sh
subsetter subset --plan-config my-plan-config.yaml --sample-config my-sample-config.yaml
```

# Sampling Multiplicity

Sampling usually means condensing a large dataset into a semantically consistent
small dataset. However, there are times that what you really want to do is
create a semantically consistent large dataset from your existing data. The
sampler has support for this by setting the multiplicity factor.

Multiplicity works by creating multiple copies of your sampled dataset in your
output database. To ensure these datasets do not collide it remaps all foreign
keys into a new key-space. Note that this process assumes your foreign keys are
opaque integers identifiers.
