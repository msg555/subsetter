import contextlib
import logging
import sys
from argparse import ArgumentParser
from typing import Optional

import yaml

from subsetter.common import DatabaseConfig
from subsetter.planner import Planner, PlannerConfig, SubsetPlan
from subsetter.sampler import (
    DirectoryOutputConfig,
    MysqlOutputConfig,
    Sampler,
    SamplerConfig,
)

LOGGER = logging.getLogger(__name__)


def _add_plan_args(parser, *, subset_action: bool = False):
    parser.add_argument(
        "--plan-config" if subset_action else "--config",
        default="planner_config.yaml",
        required=False,
        help="Path to planner config file",
        dest="plan_config",
    )
    if not subset_action:
        parser.add_argument(
            "-o",
            "--output",
            default=None,
            required=False,
            help="Output path for plan, writes to stdout by default",
            dest="plan_output",
        )


def _add_sample_args(parser, *, subset_action: bool = False):
    if not subset_action:
        parser.add_argument(
            "-p",
            "--plan",
            default="plan.yaml",
            required=False,
            help="Path to plan config file",
        )
    parser.add_argument(
        "--sample-config" if subset_action else "--config",
        default=None,
        required=False,
        help="Optional path to sample config",
        dest="sample_config",
    )
    parser.add_argument(
        "--truncate",
        action="store_const",
        const=True,
        default=False,
        help="Truncate existing output before sampling",
    )
    output_parsers = parser.add_subparsers(
        required=False,
        dest="output",
        help="Configure sampling output destination",
    )

    mysql_parser = output_parsers.add_parser(
        "mysql", help="Write sampled data to a mysql database"
    )
    mysql_parser.add_argument("--host", default=None, dest="dst_host")
    mysql_parser.add_argument("--port", type=int, default=None, dest="dst_port")
    mysql_parser.add_argument("--user", default=None, dest="dst_user")
    mysql_parser.add_argument("--password", default=None, dest="dst_password")

    directory_parser = output_parsers.add_parser(
        "directory", help="Write sampled data into directory as json files"
    )
    directory_parser.add_argument("dir", help="Path to write json output")


def _parse_args():
    parser = ArgumentParser(
        description="Database subsetter tool",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
    )
    parser.add_argument("--src-host", default=None, help="source database host")
    parser.add_argument(
        "--src-port", type=int, default=None, help="source database port"
    )
    parser.add_argument("--src-user", default=None, help="source database user")
    parser.add_argument("--src-password", default=None, help="source database password")
    subparsers = parser.add_subparsers(
        required=True,
        dest="action",
        help="Control what subsetter action to perform",
    )
    _add_plan_args(
        subparsers.add_parser("plan", help="create sampling plan from source database")
    )
    _add_sample_args(
        subparsers.add_parser("sample", help="sample a source database from a plan")
    )
    subset_parser = subparsers.add_parser("subset", help="plan and sample a database")
    _add_plan_args(subset_parser, subset_action=True)
    _add_sample_args(subset_parser, subset_action=True)
    return parser.parse_args()


def _get_source_database_config(
    args, *, overlay: Optional[DatabaseConfig] = None
) -> DatabaseConfig:
    overlay = overlay or DatabaseConfig()
    return DatabaseConfig(
        host=args.src_host or overlay.host,
        port=args.src_port or overlay.port,
        username=args.src_user or overlay.username,
        password=args.src_password or overlay.password,
    )


def _get_sample_config(args) -> SamplerConfig:
    if args.output and args.sample_config:
        LOGGER.error("Cannot pass --sample-config and output subcommand")
        sys.exit(1)
    if not args.output and not args.sample_config:
        LOGGER.error("No --sample-config or output subcommand selected")
        sys.exit(1)

    if args.output:
        if args.output == "mysql":
            return SamplerConfig(
                source=_get_source_database_config(args),
                output=MysqlOutputConfig(
                    mode="mysql",
                    host=args.dst_host,
                    port=args.dst_port,
                    username=args.dst_user,
                    password=args.dst_password,
                ),
            )
        if args.output == "directory":
            return SamplerConfig(
                source=_get_source_database_config(args),
                output=DirectoryOutputConfig(
                    mode="directory",
                    directory=args.dir,
                ),
            )
        raise RuntimeError("unexpected output subcommand")

    try:
        with open(args.sample_config, "r", encoding="utf-8") as fconfig:
            config = SamplerConfig.parse_obj(yaml.safe_load(fconfig))
    except ValueError as exc:
        LOGGER.error(
            "Unexpected sampler config file format: %s",
            exc,
            exc_info=args.verbose > 1,
        )
        sys.exit(1)
    except IOError as exc:
        LOGGER.error(
            "Could not open sampler config file %r: %s",
            args.plan_config,
            exc,
            exc_info=args.verbose > 1,
        )
        sys.exit(1)

    config.source = _get_source_database_config(args, overlay=config.source)
    return config


def _get_plan_config(args) -> PlannerConfig:
    try:
        with open(args.plan_config, "r", encoding="utf-8") as fconfig:
            return PlannerConfig.parse_obj(yaml.safe_load(fconfig))
    except ValueError as exc:
        LOGGER.error(
            "Unexpected plan file format: %s",
            exc,
            exc_info=args.verbose > 1,
        )
        sys.exit(1)
    except IOError as exc:
        LOGGER.error(
            "Could not open plan config file %r: %s",
            args.plan_config,
            exc,
            exc_info=args.verbose > 1,
        )
        sys.exit(1)


def _main_plan(args):
    config = _get_plan_config(args)
    config.source = _get_source_database_config(args, overlay=config.source)
    plan = Planner(config).plan()
    try:
        ctx = contextlib.nullcontext(sys.stdout)
        if args.plan_output:
            ctx = open(args.plan_output, "w", encoding="utf-8")
        with ctx as fplan:
            yaml.dump(
                plan.dict(exclude_defaults=True),
                stream=fplan,
                default_flow_style=False,
                width=2**20,
                sort_keys=False,
            )
    except IOError as exc:
        LOGGER.error(
            "Could not write plan to output file %r: %s", args.plan_output, exc
        )
        sys.exit(1)


def _main_sample(args):
    try:
        with open(args.plan, "r", encoding="utf-8") as fplan:
            plan = SubsetPlan.parse_obj(yaml.safe_load(fplan))
    except ValueError as exc:
        LOGGER.error(
            "Unexpected plan file format: %s",
            exc,
            exc_info=args.verbose > 1,
        )
        sys.exit(1)
    except IOError as exc:
        LOGGER.error(
            "Could not open plan file %r: %s",
            args.plan,
            exc,
            exc_info=args.verbose > 1,
        )
        sys.exit(1)

    Sampler(_get_sample_config(args)).sample(plan, truncate=args.truncate)


def _main_subset(args):
    planner_config = _get_plan_config(args)
    planner_config.source = _get_source_database_config(
        args, overlay=planner_config.source
    )
    sampler_config = _get_sample_config(args)
    sampler_config.source = planner_config.source
    plan = Planner(planner_config).plan()
    Sampler(sampler_config).sample(plan, truncate=args.truncate)


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose > 0 else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        if args.action == "plan":
            _main_plan(args)
        elif args.action == "sample":
            _main_sample(args)
        elif args.action == "subset":
            _main_subset(args)
        else:
            raise RuntimeError("Unknown action")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        LOGGER.error(
            "Unexpected error: %s",
            exc,
            exc_info=args.verbose > 1,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
