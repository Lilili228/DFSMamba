import argparse
import os
from util import util
import torch
import yaml


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        default_sssm_config = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'configs',
            'sssm.yaml',
        )
        self.parser.add_argument('--dataroot', required=False, default="/home/sda/lss/datasets/AVIID-1",help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--text_path', help='path to text file for KAIST datasets')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=512, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netD', type=str, default='DFSMambaDiscriminator', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='DFSMambaGenerator', help='selects model to use for netG')
        self.parser.add_argument('--n_layers_D', type=int, default=4, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='DFSMamba', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset_mode', type=str, default='AVIID_1', help='chooses how datasets are loaded. [unaligned | aligned | single]')
        self.parser.add_argument('--model', type=str, default='dfsmamba', help='chooses which model to use. dfsmamba, cycle_gan, pix2pix, test')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8098, help='visdom port of the web display')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument(
            '--sssm_config',
            type=str,
            default=default_sssm_config,
            help='YAML file containing sssm.* and paired augmentation defaults',
        )
        self.parser.add_argument(
            '--sssm_weight', '--lambda_sssm',
            dest='sssm_weight',
            type=float,
            default=1.0,
            help='weight for the paper SSSM objective',
        )
        self.parser.add_argument(
            '--sssm_checkpoint',
            type=str,
            help='pretrained infrared SSSM encoder checkpoint',
        )
        self.parser.add_argument(
            '--sssm_input_size',
            type=int,
            default=256,
            help='square feature-extraction size used by SSSM',
        )
        self.parser.add_argument(
            '--sssm_scale_weights',
            type=float,
            nargs=5,
            metavar=('W1', 'W2', 'W3', 'W4', 'W5'),
            default=(1.0, 0.5, 0.25, 0.125, 0.125),
            help='five multiscale SSSM feature weights',
        )
        self.parser.add_argument('--paired_random_crop', action='store_true',
                                 default=None)
        self.parser.add_argument('--paired_flip_probability', type=float)
        self.parser.add_argument('--paired_rotation_degrees', type=float)
        self.parser.add_argument('--visible_brightness_jitter', type=float)
        self.parser.add_argument('--infrared_brightness_jitter', type=float)
        self.parser.add_argument(
            '--no_dgm_dwt',
            action='store_true',
            help='disable DWT decomposition inside DGM for ablation',
        )
        self.parser.add_argument(
            '--no_multi_direction_scan',
            action='store_true',
            help='use single-direction scan ids in SS2D for ablation',
        )
        self.parser.add_argument(
            '--no_direction_gate',
            action='store_true',
            help='disable channel-direction gating in direction-adaptive SS2D',
        )

        self.initialized = True

    def _load_yaml_defaults(self):
        probe = argparse.ArgumentParser(add_help=False)
        probe.add_argument('--sssm_config')
        config_arg, _ = probe.parse_known_args()
        config_path = config_arg.sssm_config or self.parser.get_default(
            'sssm_config'
        )
        with open(config_path, 'r') as config_file:
            values = yaml.safe_load(config_file) or {}

        sssm = values.get('sssm', {})
        augmentation = values.get('augmentation', {})
        allowed_sssm = {'weight', 'checkpoint', 'input_size', 'scale_weights'}
        unknown = set(sssm) - allowed_sssm
        if unknown:
            raise ValueError('Unknown sssm config keys: %s' % sorted(unknown))
        defaults = {'sssm_' + key: value for key, value in sssm.items()}
        defaults.update(augmentation)
        self.parser.set_defaults(**defaults)

    @staticmethod
    def _validate_sssm_options(opt, is_train):
        if opt.sssm_weight < 0:
            raise ValueError('sssm.weight must be non-negative')
        if opt.sssm_input_size <= 0:
            raise ValueError('sssm.input_size must be positive')
        if len(opt.sssm_scale_weights) != 5:
            raise ValueError('sssm.scale_weights must contain five values')
        if any(weight < 0 for weight in opt.sssm_scale_weights):
            raise ValueError('sssm.scale_weights must be non-negative')
        if is_train and opt.sssm_weight > 0 and not opt.sssm_checkpoint:
            raise ValueError(
                'sssm.checkpoint is required when SSSM is enabled'
            )
        if not 0.0 <= opt.paired_flip_probability <= 1.0:
            raise ValueError('paired_flip_probability must be in [0, 1]')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self._load_yaml_defaults()
        self.opt = self.parser.parse_args()
        self._validate_sssm_options(self.opt, self.isTrain)
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
