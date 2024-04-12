# Subsetter

Subsetter is a Python utility that can be used for subsetting portions of
mysql databases. _Subsetting_ is the action extracting a smaller set of rows
from your database that still maintain expected foreign-key relationships
between your data. This can be useful for testing against a small but
realistic dataset or for generating sample data for use in demonstrations.
This tool also supports filtering that allows you to remove/anonymize rows that
may contain sensitive data.

Similar tools include Tonic.ai's platform and [condenser](https://github.com/TonicAI/condenser).
This is meant to be a simple CLI tool that overcomes many of the difficulties in
using `condenser.

## Limitations

Be aware that subsetting is a hard problem. The planner tool is meant to do a
bit of "magic" to generate a plan. For some organizations this will entirely
match their needs, for others this may only be a starting point. For this reason
the subsetter splits its function into a "planning" phase and a "sampling"
phase. The output of the planning phase can be examined and modified and fed
into the sampling phase which is responsible for the mechanics of filtering
and loading data into the destination.

Additionally the subsetter tool takes an approach of "one table, one query". This
means that the subsetter will sample each table using a single query that can
optionally reference some previously sampled rows from other tables. In
particular, this tool cannot generically sample a transitive closure of foreign
key relationships if schemas contain relationship cycles that aren't innately
closed.

# Usage

## Create a sampling plan

The first step in subsetting a database is to generate a sampling plan. A
sampling plan defines the query that will be used to sample each table

is nothing more than an ordered sequence of SQL describing how
to sample each requested database table. You'll want to create a configuration
file similar to [planner_config.example.yaml](planner_config.example.yaml) that
tells the planner what tables you want to sample along with any additional
constraints that should be considered. Then you can create a plan with the
below command:

```sh
python -m subsetter plan -c my-config.yaml > plan.yaml
```

If you inspect the generated plan file you will see SQL for each database table.
Some tables may have a `materialize: true` flag; these are sampled tables that
need to be referenced by other sampled tables. You may see some queries use a
placeholder `<SAMPLED>` identifier; this means the query is referencing the
already sampled data for that table rather than the original source table.

Generally, you can make arbitrary changes to the SQL listed in the plan with the
constraint that each query can only access the sampling results from
tables marked as _materialized_ that appear earlier in the plan.

## Sample a database with a plan

The sample sub-command will sample rows from the source database into your
target output (either a database or as json files) using a plan generated
using the plan sub-command. This tool will **not** copy schema from the source
database. Any sampled tables must already exist in the destination database.
Additionally you must pass `--truncate` if you wish to clear any existing data
in the sampled tables that may interfere with the sampling process.

```sh
export SUBSET_DESTINATION_PASSWORD=my-db-password
python -m subsetter sample --plan my-plan.yaml mysql --host my-db-host --user my-db-user
```
The sampling process proceeds in three phases:

1. If `--truncate` is specified any tables about to be sampled will be first truncated.
2. Any _materialized_ tables in the plan will be sampled into temporary tables
on the source database in the order listed in the plan file.
3. Data is copied for each table from the source to destination. The ordering
may differ from the plan file in order to adhere to foreign key constraints.

The sampler also supports filters which allow you to transform and anonymize your
data using simple column filters. See
[sampler_config.sample.yaml](sampler_config.sample.yaml) for more details on what
filters are available and how to configure them.

## Plan and sample in one action

There's also a `subset` subcommand to perform the `plan` and `sample` actions
back-to-back. This will automatically feed the generated plan into the sampler,
in addition to ensuring the same source database configuration is used for
each.

```sh
export SUBSET_DESTINATION_PASSWORD=my-db-password
python -m subsetter subset --plan-config my-config.yaml mysql --host my-db-host --user my-db-user
```

# Sampling Multiplicity

Sampling usually means condensing a large dataset into a semantically consistent
small dataset. However, there are times that what you really want to do is
create a semantically consistent large dataset from your existing data. The
sampler has support through this by setting the multiplicity factor.

Multiplicity works by creating multiple copies of your sampled dataset in your
output database. To ensure these datasets do not collide it remaps all foreign
keys into a new key-space. Note that this process assumes your foreign keys are
opaque integers identifiers.
