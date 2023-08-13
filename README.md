This project is still largely a work in progress.

Subsetter is a Python utility that can be used for subsetting portions of
mysql databases. "Subsetting" is the action extracting a smaller set of rows
from your database that still maintain expected relationships between your data.
This can be useful for testing against a small but realistic dataset or for
generating sample data for use in demonstrations.

Similar tools include Tonic.ai's platform and [condenser](https://github.com/TonicAI/condenser).
This is meant to be a simple CLI tool that overcomes many of the difficulties in
using `condenser`.

Be aware that subsetting is a hard problem. The planner tool is meant to do a
bit of "magic" to generate a plan. For some organizations this will entirely
match their needs, for others this may only be a starting point. The plan
produced can be fairly aribtrarily modified and then fed to the sampler which
does the technical work of actually extracting data from the source.

# Usage

## Create a sampling plan

The first step in subsetting a database is to generate a sampling plan. A
sampling plan is nothing more than an ordered sequence of SQL describing how
to sample each requested database table. You'll want to create a configuration
file similar to [planner_config.sample.yaml](planner_config.sample.yaml) that
tells the planner what tables you want to sample. Then you can create a plan
with the below command:

```sh
python -m subsetter plan -c my-config.yaml > plan.yaml
```

If you inspect the generated plan file you will see SQL for each database table.
Some tables may have a `materialize: true` flag; these are sampled tables that
need to be referenced by other sampled tables. You may see some queries use a
placeholder `<SAMPLED>` identifier; this means the query is referencing the
already sampled data for that table rather than the original source table.


## Sample a database with a plan

The sample sub-command will sample rows from the source database into your
target output (either a database or as json files) using a plan generated
using the plan sub-command.

```sh
export SUBSET_DESTINATION_PASSWORD=my-db-password
python -m subsetter sample --plan my-plan.yaml mysql --host my-db-host --user my-db-user
```

## Plan and sample in one action

There's also a `subset` subcommand to perform the `plan` and `sample` actions
back-to-back. This will automatically feed the generated plan into the sampler,
in addition to ensuring the same source database configuration is used for
each.

```sh
export SUBSET_DESTINATION_PASSWORD=my-db-password
python -m subsetter subset --plan-config my-config.yaml mysql --host my-db-host --user my-db-user
```

# Future Work

This project is still relatively incomplete and lacks some basic things like:

- More complete documentation
- Testing

Additionally the following ideas are potential future work for features:

- Data anonymization, ideally without ever commiting unanonymized data to destination
