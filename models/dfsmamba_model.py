import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from ssim import SSIM
import time
from .dfsmamba import architecture, configs
from .sssm import SSSMLoss

class DFSMambaModel(BaseModel):
    def name(self):
        return 'DFSMamba'

    @staticmethod
    def _apply_ablation_options(config, opt):
        config.use_dgm_dwt = not opt.no_dgm_dwt
        config.mamba.use_multi_direction_scan = (
            not opt.no_multi_direction_scan
        )
        config.mamba.use_direction_gate = not opt.no_direction_gate

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain  # or True,
        if self.isTrain:
            gen_config = configs.get_train_gen_config()
            disc_config = configs.get_train_disc_config()
        else:
            gen_config = configs.get_test_config()
            disc_config = None

        self._apply_ablation_options(gen_config, opt)
        if disc_config is not None:
            self._apply_ablation_options(disc_config, opt)

        # load/define networks
        runtime_device = (
            torch.device('cuda:%d' % self.gpu_ids[0])
            if self.gpu_ids else torch.device('cpu')
        )
        self.runtime_device = runtime_device
        self.netG = architecture.DFSMambaGenerator(gen_config).to(runtime_device)
        if self.isTrain:
            self.netD = architecture.DFSMambaDiscriminator(disc_config).to(
                runtime_device
            )
        if not self.isTrain or opt.continue_train:
            print(f'Testing ====>{opt.which_epoch}')
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:  # or True:
                self.load_network(self.netD, 'D', opt.which_epoch)

        self.fake_AB_pool = ImagePool(opt.pool_size)
        # define loss functions
        self.criterionGAN = networks.GANLoss(
            use_lsgan=not opt.no_lsgan,
            tensor=self.Tensor,
        )
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionMSE = torch.nn.MSELoss()
        self.ssim = SSIM(val_range=2.0, input_min=-1.0)
        self.use_sssm = bool(self.isTrain and opt.sssm_weight > 0)
        if self.use_sssm:
            self.criterionSSSM = SSSMLoss(
                encoder_weights=opt.sssm_checkpoint,
                lambda_weight=opt.sssm_weight,
                scale_weights=opt.sssm_scale_weights,
                input_size=opt.sssm_input_size,
                device=runtime_device,
            ).to(runtime_device)

        # initialize optimizers
        if self.isTrain:
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(),
                lr=opt.lr / 100,
                betas=(opt.beta1, 0.999),
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if self.isTrain:
            print('---------- Networks initialized -------------')
            networks.print_network(self.netG)
            networks.print_network(self.netD)
            print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], non_blocking=True)
            input_B = input_B.cuda(self.gpu_ids[0], non_blocking=True)
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.fake_B = self.netG(self.real_A)

    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.real_A = Variable(self.input_A)
            self.real_B = Variable(self.input_B)
            t = time.time()
            self.fake_B = self.netG(self.real_A)
            t = time.time() - t
            self.real_B = Variable(self.input_B)
        return t


    # get image paths
    def get_image_paths(self):
        return self.image_paths


    def backward_D(self):
        if torch.isnan(self.fake_B).any():
            print("fake_B contains NaN before backward_D!")
            exit()
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(
            torch.cat((self.real_A, self.fake_B), 1).data
        )
        # pred_fake, pred_middle_fake = self.netD(fake_AB.detach())
        (
            pred_fake,
            tf_middle_fake,
            vm_middle_fake,
            diffs_fake,
            prods_fake,
        ) = self.netD(fake_AB.detach())
        self.loss_D_fake = ((self.criterionGAN(pred_fake, False)
                             + self.criterionGAN(tf_middle_fake, False))
                            + self.criterionGAN(vm_middle_fake, False))
        # self.loss_mse_dp_fake = -(self.cal_mse(diffs_fake) - self.cal_mse(prods_fake))
        for i in range(len(diffs_fake)):
            self.loss_D_fake += self.criterionGAN(diffs_fake[i], False)
            self.loss_D_fake += self.criterionGAN(prods_fake[i], False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        # pred_real, pred_middle_real = self.netD(real_AB)
        (
            pred_real,
            tf_middle_real,
            vm_middle_real,
            diffs_real,
            prods_real,
        ) = self.netD(real_AB)
        self.loss_D_real = (self.criterionGAN(pred_real, True)
                            + self.criterionGAN(tf_middle_real, True)
                            + self.criterionGAN(vm_middle_real, True))
        # self.loss_mse_dp_real = (self.cal_mse(diffs_real) - self.cal_mse(prods_real))
        for i in range(len(diffs_real)):
            self.loss_D_real += self.criterionGAN(diffs_real[i], True)
            self.loss_D_real += self.criterionGAN(prods_real[i], True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        # self.loss_D = (
        #     self.loss_D_fake + self.loss_D_real
        #     + self.loss_mse_dp_fake + self.loss_mse_dp_real
        # ) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        if torch.isnan(self.fake_B).any():
            print("fake_B contains NaN before backward_D!")
            exit()

        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        out, tf_middle_out, vm_middle_out, diffs, prods = self.netD(fake_AB)
        with torch.autograd.set_detect_anomaly(True):
            # GAN Loss
            self.loss_G_GAN = (self.criterionGAN(out, True) +
                               self.criterionGAN(tf_middle_out, True) +
                               self.criterionGAN(vm_middle_out, True))


            self.loss_G_L1 = (
                self.criterionL1(self.fake_B, self.real_B)
                * self.opt.lambda_A
            )
            self.loss_ssim = (
                1 - self.ssim(self.fake_B, self.real_B)
            ) * self.opt.lambda_A

            if self.use_sssm:
                self.loss_G_SSSM = self.criterionSSSM(
                    self.real_B,
                    self.fake_B,
                )
            else:
                self.loss_G_SSSM = torch.zeros(
                    (),
                    device=self.fake_B.device,
                )

            self.loss_G = (
                self.loss_G_GAN
                + self.loss_G_L1
                + self.loss_ssim
                + self.loss_G_SSSM
            )

            self.loss_G.backward()

    @staticmethod
    def set_requires_grad(network, requires_grad):
        for parameter in network.parameters():
            parameter.requires_grad_(requires_grad)

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad(set_to_none=True)
        self.backward_D()
        self.optimizer_D.step()
        self.optimizer_D.zero_grad(set_to_none=True)

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad(set_to_none=True)
        self.backward_G()
        self.optimizer_G.step()
        self.set_requires_grad(self.netD, True)

    def get_current_errors(self):
        errors = OrderedDict([
            ('G_GAN', self.loss_G_GAN.detach()),
            ('G_L1', self.loss_G_L1.detach()),
            ('G_SSIM', self.loss_ssim.detach()),
            ('D_real', self.loss_D_real.detach()),
            ('D_fake', self.loss_D_fake.detach()),
        ])
        if self.use_sssm:
            errors['G_SSSM'] = self.loss_G_SSSM.detach()
        return errors

    def get_errors(self):
        errors = OrderedDict([
            ('G_GAN', 0),
            ('G_L1', 0),
            ('G_SSIM', 0),
            ('D_real', 0),
            ('D_fake', 0),
        ])
        if self.use_sssm:
            errors['G_SSSM'] = 0
        return errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        # fake_B = util.thermal_tensor2im(self.fake_B.data)
        # real_B = util.thermal_tensor2im(self.real_B.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)


    def cal_mse(self, flist):
        res = 0
        for tensor in flist:
            tensor_mean = torch.mean(torch.tensor(tensor))
            res += tensor_mean
        return res
