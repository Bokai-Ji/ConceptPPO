from common.env.procgen_wrappers import *
from common.logger import Logger
from common.storage import Storage
from common.model import NatureModel, ImpalaModel, NoneModel, ConceptImpalaModel, MLP
from common.policy import CategoricalPolicy, ConceptPolicy
from common import set_global_seeds, set_global_log_levels

import os, time, yaml, argparse
import gym
from procgen import ProcgenEnv
import random
import torch
from distutils.util import strtobool


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',         type=str, default = 'test', help='experiment name')
    parser.add_argument('--env_name',         type=str, default = 'coinrun', help='environment ID')
    parser.add_argument('--start_level',      type=int, default = int(0), help='start-level for environment')
    parser.add_argument('--num_levels',       type=int, default = int(0), help='number of training levels for environment')
    parser.add_argument('--distribution_mode',type=str, default = 'easy', help='distribution mode for environment')
    parser.add_argument('--param_name',       type=str, default = 'easy-200', help='hyper-parameter ID')
    parser.add_argument('--device',           type=str, default = 'gpu', required = False, help='whether to use gpu')
    parser.add_argument('--gpu_device',       type=int, default = int(0), required = False, help = 'visible device in CUDA')
    parser.add_argument('--num_timesteps',    type=int, default = int(25000000), help = 'number of training timesteps')
    parser.add_argument('--seed',             type=int, default = random.randint(0,9999), help='Random generator seed')
    parser.add_argument('--log_level',        type=int, default = int(40), help='[10,20,30,40]')
    parser.add_argument('--num_checkpoints',  type=int, default = int(1), help='number of checkpoints to store')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True, help='if toggled, this experiment will be tracked with wandb')
    parser.add_argument('--wandb_project_name', type=str, default='ConceptPPO', help='the wandb`s project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='the entity (team) of wandb`s project')

    args = parser.parse_args()
    exp_name = args.exp_name
    env_name = args.env_name
    start_level = args.start_level
    num_levels = args.num_levels
    distribution_mode = args.distribution_mode
    param_name = args.param_name
    device = args.device
    gpu_device = args.gpu_device
    num_timesteps = args.num_timesteps
    seed = args.seed
    log_level = args.log_level
    num_checkpoints = args.num_checkpoints

    set_global_seeds(seed)                      #TODO Set Seeds
    set_global_log_levels(log_level)    #TODO Set Log Levels

    ####################
    ## HYPERPARAMETERS #
    ####################
    #TODO Loading Hyperparameters from `config.yml`
    run_name = f"{args.env_name}_{args.exp_name}_{args.seed}_{int(time.time())}"
    print('[LOADING HYPERPARAMETERS...]')
    with open('./hyperparams/procgen/config.yml', 'r') as f:
        hyperparameters = yaml.safe_load(f)[param_name]         
    for key, value in hyperparameters.items():
        print(key, ':', value)

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config={**hyperparameters, **vars(args)},
            name=run_name,
            save_code=True
        )

    ############
    ## DEVICE ##
    ############
    #TODO Set Training Device, seems only support 1 GPU at a run
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    device = torch.device('cuda')

    #################
    ## ENVIRONMENT ##
    #################
    #TODO  Set Environment, note that we
    #TODO  - Disabled the background
    #TODO  - Restrict Themes so that the style will not change during training
    #TODO  - Use-monochrome_assets so that the model can focus on digging concepts and relational features
    print('INITIALIZAING ENVIRONMENTS...')
    n_steps = hyperparameters.get('n_steps', 256)
    n_envs = hyperparameters.get('n_envs', 64)
    # By default, pytorch utilizes multi-threaded cpu
    # Procgen is able to handle thousand of steps on a single core
    torch.set_num_threads(1)
    env = ProcgenEnv(num_envs=n_envs,
                     env_name=env_name,
                     start_level=start_level,
                     num_levels=num_levels,
                     distribution_mode=distribution_mode,
                     use_backgrounds = False,
                     restrict_themes= True,
                     use_monochrome_assets = True)
    normalize_rew = hyperparameters.get('normalize_rew', True)
    env = VecExtractDictObs(env, "rgb") #TODO This wrapper do observation['rgb'] to extract observation from obs_dict
    if normalize_rew:
        env = VecNormalize(env, ob=False) #TODO Normalizing returns, but not the img frames. This trick is mentioned in champion's blog as the only important trick from ALE envs.
    env = TransposeFrame(env) #TODO Transpose frames from [W, H, C] to [C, W, H]
    # env = TBlobPROS(env, basic=False) #TODO Our PROS wrapper
    env = ScaledFloatFrame(env) #TODO `observation/255.0`

    ############
    ## LOGGER ##
    ############
    #TODO Logger
    print('INITIALIZAING LOGGER...')
    # logdir = 'procgen/' + env_name + '/' + exp_name + '/' + 'seed' + '_' + \
    #          str(seed) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = run_name
    logdir = os.path.join('logs', logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    logger = Logger(n_envs, logdir)

    ###########
    ## MODEL ##
    ###########
    #TODO get observation space, observation shape and embedder(model). We do not conside recurrent here, and policy is PPO only
    print('INTIALIZING MODEL...')
    observation_space = env.observation_space
    observation_shape = observation_space.shape
    architecture = hyperparameters.get('architecture', 'impala')
    in_channels = observation_shape[0]
    action_space = env.action_space

    # Model architecture
    if architecture == 'nature':
        model = NatureModel(in_channels=in_channels)
    elif architecture == 'impala':
        model = ImpalaModel(in_channels=in_channels)
    elif architecture == 'concept_impala':
        model = ConceptImpalaModel(in_channels=in_channels)
    elif architecture == 'mlp':
        model = MLP()
    elif architecture == 'none':
        model = NoneModel(output_dim=env.observation_space.shape[-1])

    # Discrete action space
    recurrent = hyperparameters.get('recurrent', False)
    embedding_size = hyperparameters.get('embedding_size', 32)
    n_heads = hyperparameters.get('n_heads', 2)
    ratio = hyperparameters.get('ratio', 1.9)
    method = hyperparameters.get('method', 'product')
    mode = hyperparameters.get('mode', 'ctx')
    shared_head = hyperparameters.get('shared_head', True)
    skip = hyperparameters.get('skip', 0) # No classification token
    rpe_on = hyperparameters.get('rpe_on', 'qkv')
    add_relative = hyperparameters.get('add_relative', True)
    if isinstance(action_space, gym.spaces.Discrete):
        action_size = action_space.n
        if args.param_name == 'concept-easy-200':
            policy = ConceptPolicy(embedder=model,
                                    recurrent=recurrent,
                                    action_size=action_size,
                                    n_heads=n_heads,
                                    ratio=ratio,
                                    method=method,
                                    mode=mode,
                                    shared_head=shared_head,
                                    skip=skip,
                                    rpe_on=rpe_on,
                                    add_relative=add_relative
                                    )
        elif args.param_name == 'easy-200':
            policy = CategoricalPolicy(embedder=model,
                                        recurrent=recurrent,
                                        action_size=action_size)
    else:
        raise NotImplementedError
    policy.to(device)

    #############
    ## STORAGE ##
    #############
    #TODO Storage for PPO roll-out steps
    print('INITIALIZAING STORAGE...')
    hidden_state_dim = model.output_dim
    storage = Storage(observation_shape, hidden_state_dim, n_steps, n_envs, device)

    ###########
    ## AGENT ##
    ###########
    #TODO Building agents
    print('INTIALIZING AGENT...')
    algo = hyperparameters.get('algo', 'ppo')
    if algo == 'ppo':
        from agents.ppo import PPO as AGENT
    else:
        raise NotImplementedError
    agent = AGENT(env, policy, logger, storage, device, num_checkpoints, **hyperparameters)

    ##############
    ## TRAINING ##
    ##############
    #TODO Start training
    print('START TRAINING...')
    agent.train(num_timesteps, args.param_name)
