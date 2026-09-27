"""Small explicit CLI with stable JSON output and meaningful exit codes."""
import argparse
import json
import sys
from . import __version__
from .contracts import load_json, validate_task, default_policy
from .budget import BudgetExceeded


def _parser():
    parser = argparse.ArgumentParser(prog='ctb', description='Independent crystal task evaluation: native analysis, authorized MLFF recipes, and DFT planning. Registration does not imply live verification.')
    parser.add_argument('--version', action='version', version=f'ctb {__version__}')
    commands = parser.add_subparsers(dest='command', required=True)
    doctor = commands.add_parser('doctor', help='Report interface, installation, configuration and verification separately')
    doctor.add_argument('--deployment')
    validate = commands.add_parser('validate-task', help='Validate a data-only v2 JSON task')
    validate.add_argument('path', nargs='?')
    validate.add_argument('--task')
    for verb in ('plan', 'evaluate'):
        cmd = commands.add_parser(verb, help='Plan only' if verb == 'plan' else 'Evaluate native tasks, external DFT exchange, or explicitly authorized local backends')
        cmd.add_argument('--task', required=True)
        cmd.add_argument('--structures', nargs='+', required=True)
        cmd.add_argument('--output', required=True)
        cmd.add_argument('--backend', choices=['auto','mlff','dft','hybrid'])
        cmd.add_argument('--policy')
        cmd.add_argument('--protocol')
        cmd.add_argument('--deployment')
        cmd.add_argument('--mlff-provider')
        cmd.add_argument('--dft-provider')
        cmd.add_argument('--cache')
        cmd.add_argument('--dry-run', action='store_true')
    dft = commands.add_parser('dft', help='Installed adapters and engine-free file exchange')
    verbs = dft.add_subparsers(dest='dft_command', required=True)
    verbs.add_parser('adapters', help='Read installed static manifests without importing plugins')
    prep = verbs.add_parser('prepare', help='Export only the current geometry frontier; never launch an engine')
    prep.add_argument('--structures', nargs='+')
    prep.add_argument('--task')
    prep.add_argument('--site')
    prep.add_argument('--adapter')
    prep.add_argument('--protocol')
    prep.add_argument('--output', required=True)
    coll = verbs.add_parser('collect', help='Import outputs and score with the existing CTB assessor')
    coll.add_argument('--run', required=True)
    coll.add_argument('--results')
    coll.add_argument('--output')
    local = verbs.add_parser('run-local', help='Explicitly authorized thin local launch; never downloads an engine')
    local.add_argument('--request-directory', required=True)
    local.add_argument('--site', required=True)
    local.add_argument('--ledger-directory', required=True)
    reass = verbs.add_parser('reassess', help='Change task thresholds without engine calls')
    reass.add_argument('--run', required=True)
    reass.add_argument('--task', required=True)
    reass.add_argument('--output', required=True)
    summary = commands.add_parser('summarize', help='Read an existing metrics artifact without recomputation')
    summary.add_argument('path')
    return parser


def _policy(args):
    policy = load_json(args.policy) if args.policy else None
    explicit = {'mode': args.backend, 'preferred_mlff': args.mlff_provider, 'preferred_dft': args.dft_provider}
    if policy:
        for key, value in explicit.items():
            if value is not None and key in policy and policy[key] != value:
                raise ValueError(f'conflicting explicit CLI/file policy field: {key}')
    if policy is None and any(value is not None for value in explicit.values()):
        policy = default_policy(args.backend or 'auto')
        policy.pop('preferred_mlff', None)
        policy.pop('preferred_dft', None)
    if policy is not None:
        for key, value in explicit.items():
            if value is not None:
                policy[key] = value
    return policy


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.command == 'dft':
            from .dft import exchange
            from .dft.registry import discover
            if args.dft_command == 'adapters':
                result = {'adapters': discover(), 'workers_executed': 0}
            elif args.dft_command == 'prepare':
                result = exchange.prepare(args.structures, args.task, output=args.output, site=args.site,
                                          adapter=args.adapter, protocol=args.protocol)
            elif args.dft_command == 'collect':
                result = exchange.collect(args.run, args.results, output=args.output)
            elif args.dft_command == 'run-local':
                from .dft.local import run_local
                result = run_local(args.request_directory, load_json(args.site), args.ledger_directory)
            else:
                result = exchange.reassess(args.run, args.task, output=args.output)
        elif args.command == 'doctor':
            from .backends import doctor
            result = doctor(load_json(args.deployment) if args.deployment else None)
        elif args.command == 'validate-task':
            path = args.task or args.path
            if not path:
                raise ValueError('provide --task or a task path')
            task = validate_task(load_json(path))
            result = {'status':'valid', 'task_id':task['task_id'], 'schema_version':task['schema_version']}
        elif args.command == 'summarize':
            from .api import summarize
            result = summarize(args.path)
        else:
            from .api import evaluate
            result = evaluate(args.structures, args.task, output=args.output, policy=_policy(args),
                              protocol=args.protocol, deployment=args.deployment,
                              dry_run=args.dry_run or args.command == 'plan', cache_root=args.cache)
            print(json.dumps(result.get('metrics', result['plan']), ensure_ascii=False, indent=2, allow_nan=False))
            return 0 if result['plan']['status'] == 'ready' else 3
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if args.command == 'dft' and result.get('plan',{}).get('status','ready') != 'ready':return 3
        return 0
    except (ValueError, OSError, BudgetExceeded) as exc:
        print(json.dumps({'status':'error', 'error':str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
