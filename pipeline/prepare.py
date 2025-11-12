from argparse import Namespace
from datetime import datetime
from loguru import logger

from util.util import set_seed
def prepare(args: Namespace, is_main: bool):
    set_seed(args.seed)
    
    cur_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    args.task_name = f'{cur_time}-{args.dataset}-{args.model}-{args.tag}'
    logger.add(f'log/{args.task_name}/.log') if is_main else None
    if args.w_mse == 0:
        args.ensemble = False
        logger.warning(f"Ensemble is disabled because w_mse = {args.w_mse}.") if is_main else None
    if args.ckpt is not None or args.model == "zero-shot":
        args.save = False
        logger.warning(f"Save is disabled because ckpt is not None or model is zero-shot.") if is_main else None
    logger.info(args) if is_main else None
    return args