from rl_games.common.algo_observer import AlgoObserver

class WandbAlgoObserver(AlgoObserver):
    """Need this to propagate the correct experiment name after initialization."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def __del__(self):
        import wandb
        wandb.finish()

    def before_init(self, base_name, config, experiment_name):
        """
        Must call initialization of Wandb before RL-games summary writer is initialized, otherwise
        sync_tensorboard does not work.
        """

        import wandb
        cfg = self.cfg

        from datetime import datetime
        experiment_name = '{date:%m%d-%H%M}_'.format(date=datetime.now()) + cfg["wandb_run_name"]
        wandb_unique_id = f"uid_{experiment_name}"
        print(f"Wandb using unique id {wandb_unique_id}")

        # this can fail occasionally, so we try a couple more times
        @retry(3, exceptions=(Exception,))
        def init_wandb():
            wandb.init(
                project=cfg["wandb_project"],
                entity=cfg["wandb_entity"],
                group=cfg["wandb_group"],
                tags=cfg["wandb_tags"],
                sync_tensorboard=True,
                id=wandb_unique_id,
                name=experiment_name,
                resume=True,
                settings=wandb.Settings(start_method='fork'),
            )
            # import os 
            # if cfg.wandb_project == 'TocabiAMPLower' or cfg.wandb_project == 'SN_weight_and_bias' or cfg.wandb_project == 'SN_paper_experiment':
            #     print('Saving files for TocabiAMPLower.....')
            #     wandb.save(os.path.join(os.getcwd(), 'isaacgymenvs/cfg/task/TocabiAMPLower.yaml'), policy='now')
            #     wandb.save(os.path.join(os.getcwd(), 'isaacgymenvs/cfg/train/TocabiAMPLowerPPO.yaml'), policy='now')
            #     wandb.save(os.path.join(os.getcwd(), 'isaacgymenvs/tasks/amp/tocabi_amp_lower_base.py'), policy='now')
            #     wandb.save(os.path.join(os.getcwd(), 'isaacgymenvs/tasks/tocabi_amp_lower.py'), policy='now')
            #     wandb.save(os.path.join(os.getcwd(), 'assets/amp/tocabi_motions/tocabi_motions.yaml'), policy='now')
            #     wandb.save(os.path.join(os.getcwd(), 'isaacgymenvs/learning/amp_continuous.py'), policy='now')
            #     wandb.save(os.path.join(os.getcwd(), 'isaacgymenvs/learning/amp_network_builder.py'), policy='now')
            # elif cfg.wandb_project == 'TocabiAMPFeet':
            #     print('Saving files for TocabiAMPFeet.....')
            #     wandb.save(os.path.join(os.getcwd(), 'isaacgymenvs/cfg/task/TocabiAMPFeet.yaml'), policy='now')
            #     wandb.save(os.path.join(os.getcwd(), 'isaacgymenvs/cfg/train/TocabiAMPFeetPPO.yaml'), policy='now')
            #     wandb.save(os.path.join(os.getcwd(), 'isaacgymenvs/tasks/amp/tocabi_amp_feet_base.py'), policy='now')
            #     wandb.save(os.path.join(os.getcwd(), 'isaacgymenvs/tasks/tocabi_amp_feet.py'), policy='now')
            #     wandb.save(os.path.join(os.getcwd(), 'assets/amp/tocabi_motions/tocabi_motions_feet.yaml'), policy='now')
            #     wandb.save(os.path.join(os.getcwd(), 'isaacgymenvs/learning/feet_amp/feet_amp_continuous.py'), policy='now')
            # elif cfg.wandb_project == 'TocabimAMPLower':
            #     print('Saving files for TocabimAMPLower.....')
            #     wandb.save(os.path.join(os.getcwd(), 'isaacgymenvs/cfg/task/TocabimAMPLower.yaml'), policy='now')
            #     wandb.save(os.path.join(os.getcwd(), 'isaacgymenvs/cfg/train/TocabimAMPLowerPPO.yaml'), policy='now')
            #     wandb.save(os.path.join(os.getcwd(), 'isaacgymenvs/tasks/amp/tocabi_mamp_lower_base.py'), policy='now')
            #     wandb.save(os.path.join(os.getcwd(), 'isaacgymenvs/tasks/amp/tocabi_mamp_lower.py'), policy='now')
            #     wandb.save(os.path.join(os.getcwd(), 'assets/amp/tocabi_motions/tocabi_motions_class.yaml'), policy='now')
            #     wandb.save(os.path.join(os.getcwd(), 'isaacgymenvs/learning/mamp_continuous.py'), policy='now') 
       
            if cfg["wandb_logcode_dir"]:
                wandb.run.log_code(root=cfg["wandb_logcode_dir"])
                print('wandb running directory........', wandb.run.dir)

        print('Initializing WandB...')
        try:
            init_wandb()
        except Exception as exc:
            print(f'Could not initialize WandB! {exc}')

        if isinstance(self.cfg, dict):
            wandb.config.update(self.cfg, allow_val_change=True)
        else:
            pass


import time
def retry(times, exceptions):
    """
    Retry Decorator https://stackoverflow.com/a/64030200/1645784
    Retries the wrapped function/method `times` times if the exceptions listed
    in ``exceptions`` are thrown
    :param times: The number of times to repeat the wrapped function/method
    :type times: Int
    :param exceptions: Lists of exceptions that trigger a retry attempt
    :type exceptions: Tuple of Exceptions
    """
    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    print(f'Exception thrown when attempting to run {func}, attempt {attempt} out of {times}')
                    time.sleep(min(2 ** attempt, 30))
                    attempt += 1

            return func(*args, **kwargs)
        return newfn
    return decorator
