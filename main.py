
import time
from accelerate import Accelerator
from loguru import logger

from pipeline.args import parse_args
from pipeline.train import Runner
from pipeline.prepare import prepare



def main():
    start = time.time()
    args = parse_args()
    accelerator = Accelerator()
    prepare(args, accelerator.is_main_process)
    runner = Runner(args, accelerator)
    if args.model == "zero-shot" or args.ckpt is not None:
        runner.eval()
    else:
        runner.run()

    if accelerator.is_main_process:
        logger.info(f"Mission complete, time: {time.time() - start:.2f}s")
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == '__main__':
    main()