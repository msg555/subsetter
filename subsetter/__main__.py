import contextlib
import logging
import sys
from argparse import ArgumentParser
from typing import Any

import yaml

from subsetter.config_model import SubsetterConfig
from subsetter.plan_model import SubsetPlan
from subsetter.planner import Planner
from subsetter.sampler import Sampler

try:
    from tqdm.contrib.logging import logging_redirect_tqdm  # type: ignore
except ImportError:
    logging_redirect_tqdm = contextlib.nullcontext

LOGGER = logging.getLogger(__name__)


@contextlib.contextmanager
def _open_config_path(path: str):
    if path == "-":
        yield sys.stdin
    else:
        with open(path, "r", encoding="utf-8") as fconfig:
            yield fconfig


def _add_plan_args(parser, *, subset_action: bool = False):
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
        "--truncate",
        action="store_const",
        const=True,
        default=False,
        help="Truncate existing output before sampling",
    )
    parser.add_argument(
        "--create",
        action="store_const",
        const=True,
        default=False,
        help="Create tables in destination from source if missing",
    )


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
    parser.add_argument(
        "-c",
        "--config",
        action="append",
        help="Path to subsetter config file, defaults to 'subsetter.yaml'",
    )
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


def _get_config(args) -> SubsetterConfig:
    config_files = args.config
    if not config_files:
        config_files = ["subsetter.yaml"]

    def _dict_merge(lhs, rhs):
        if isinstance(lhs, dict) and isinstance(rhs, dict):
            result = dict(lhs)
            for key, val in rhs.items():
                if key in result:
                    result[key] = _dict_merge(result[key], val)
                else:
                    result[key] = val
            return result
        return rhs

    try:
        config_data: Any = {}
        for config_file in config_files:
            try:
                with _open_config_path(config_file) as fconfig:
                    config_data = _dict_merge(config_data, yaml.safe_load(fconfig))
            except IOError as exc:
                LOGGER.error(
                    "Could not open subsetter config file %r: %s",
                    config_file,
                    exc,
                    exc_info=args.verbose > 1,
                )
                sys.exit(1)
        return SubsetterConfig.model_validate(config_data)
    except ValueError as exc:
        LOGGER.error(
            "Unexpected subsetter config file format: %s",
            exc,
            exc_info=args.verbose > 1,
        )
        sys.exit(1)


def _main_plan(args):
    config = _get_config(args)
    if config.planner is None:
        LOGGER.error("Config file must include a 'planner' section to run planner")
        sys.exit(1)

    plan = Planner(config.source, config.planner).plan()
    try:
        ctx = contextlib.nullcontext(sys.stdout)
        if args.plan_output:
            ctx = open(args.plan_output, "w", encoding="utf-8")
        with ctx as fplan:
            yaml.dump(
                plan.dict(exclude_unset=True, by_alias=True),
                stream=fplan,
                default_flow_style=False,
                width=2**20,
                sort_keys=True,
            )
    except IOError as exc:
        LOGGER.error(
            "Could not write plan to output file %r: %s", args.plan_output, exc
        )
        sys.exit(1)


def _main_sample(args):
    config = _get_config(args)
    try:
        with _open_config_path(args.plan) as fplan:
            plan = SubsetPlan.model_validate(yaml.safe_load(fplan))
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

    Sampler(config.source, config.sampler).sample(
        plan,
        truncate=args.truncate,
        create=args.create,
    )


def _main_subset(args):
    config = _get_config(args)
    if config.planner is None:
        LOGGER.error(
            "Config file must include a 'planner' section to run subset command"
        )
        sys.exit(1)

    plan = Planner(config.source, config.planner).plan()
    Sampler(config.source, config.sampler).sample(
        plan,
        truncate=args.truncate,
        create=args.create,
    )


def main():
    args = _parse_args()
    logging_format = "%(levelname)s\t%(message)s"
    if args.verbose > 0:
        logging_format = "%(asctime)s\t%(levelname)s\t%(message)s"
    logging.basicConfig(
        level=logging.DEBUG if args.verbose > 0 else logging.INFO,
        format=logging_format,
    )
    logging.getLogger("faker").setLevel(logging.INFO)

    with logging_redirect_tqdm():
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
                "%s: %s",
                type(exc).__name__,
                exc,
                exc_info=args.verbose > 1,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
