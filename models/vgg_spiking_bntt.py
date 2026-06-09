import torch
import torch.nn as nn
import torch.nn.functional as F


class Surrogate_BP_Function(torch.autograd.Function):
    """Rectangular surrogate gradient used for spike backpropagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(inp), 0.0, 0.0)
        return grad

def PoissonGen(inp, rescale_fac: float = 2.0):
    """Generate signed Poisson spikes from normalized image input."""
    return (torch.rand_like(inp) * rescale_fac <= torch.abs(inp)).float() * torch.sign(inp)


class SNN_VGG9_BNTT(nn.Module):
    """VGG9-style SNN with batch normalization through time (BNTT)."""

    def __init__(self, timesteps=20, leak_mem=0.995, img_size=32, num_cls=10, input_channels=3):
        super().__init__()

        self.img_size = img_size
        self.num_cls = num_cls
        self.input_channels = int(input_channels)
        self.timesteps = int(timesteps)
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = float(leak_mem)
        self.batch_num = self.timesteps

        print(">>>>>>>>>>>>>>>>>>>> VGG 9 >>>>>>>>>>>>>>>>>>>>>>")
        print(f"***** time steps per batchnorm: {self.batch_num}")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False

        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList(
            [nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(self.batch_num)]
        )
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.ModuleList(
            [nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(self.batch_num)]
        )
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt3 = nn.ModuleList(
            [nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(self.batch_num)]
        )
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt4 = nn.ModuleList(
            [nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(self.batch_num)]
        )
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt5 = nn.ModuleList(
            [nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(self.batch_num)]
        )
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt6 = nn.ModuleList(
            [nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(self.batch_num)]
        )
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt7 = nn.ModuleList(
            [nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(self.batch_num)]
        )
        self.pool3 = nn.AvgPool2d(kernel_size=2)

        self.fc1 = nn.Linear((self.img_size // 8) * (self.img_size // 8) * 256, 1024, bias=bias_flag)
        self.bntt_fc = nn.ModuleList(
            [nn.BatchNorm1d(1024, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(self.batch_num)]
        )
        self.fc2 = nn.Linear(1024, self.num_cls, bias=bias_flag)

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7]
        self.bntt_list = [self.bntt1, self.bntt2, self.bntt3, self.bntt4, self.bntt5, self.bntt6, self.bntt7, self.bntt_fc]
        self.pool_list = [False, self.pool1, False, self.pool2, False, False, self.pool3]

        # for bn_list in self.bntt_list:
            # for bn in bn_list:
                # bn.bias = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=2)
            elif isinstance(m, nn.Linear):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=2)

    def forward(self, inp: torch.Tensor, return_time_outputs: bool = False):
        """Run the spiking forward pass and optionally return per-timestep logits."""
        device = inp.device
        batch_size = inp.size(0)

        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size, device=device)
        mem_conv2 = torch.zeros(batch_size, 64, self.img_size, self.img_size, device=device)
        mem_conv3 = torch.zeros(batch_size, 128, self.img_size // 2, self.img_size // 2, device=device)
        mem_conv4 = torch.zeros(batch_size, 128, self.img_size // 2, self.img_size // 2, device=device)
        mem_conv5 = torch.zeros(batch_size, 256, self.img_size // 4, self.img_size // 4, device=device)
        mem_conv6 = torch.zeros(batch_size, 256, self.img_size // 4, self.img_size // 4, device=device)
        mem_conv7 = torch.zeros(batch_size, 256, self.img_size // 4, self.img_size // 4, device=device)
        mem_conv_list = [mem_conv1, mem_conv2, mem_conv3, mem_conv4, mem_conv5, mem_conv6, mem_conv7]

        mem_fc1 = torch.zeros(batch_size, 1024, device=device)
        mem_fc2 = torch.zeros(batch_size, self.num_cls, device=device)
        time_outputs = [] if return_time_outputs else None

        for t in range(self.timesteps):
            #spike_inp = PoissonGen(inp)
            spike_inp = PoissonGen(inp, rescale_fac=6.0 * (self.timesteps / 30.0))

            out_prev = spike_inp

            for i, conv in enumerate(self.conv_list):
                mem = self.leak_mem * mem_conv_list[i] + self.bntt_list[i][t](conv(out_prev))

                mem_thr = (mem / conv.threshold) - 1.0
                out = self.spike_fn(mem_thr)

                rst = torch.zeros_like(mem)
                rst[mem_thr > 0] = conv.threshold
                mem = mem - rst

                mem_conv_list[i] = mem
                out_prev = out

                if self.pool_list[i] is not False:
                    out_prev = self.pool_list[i](out_prev)

            out_prev = out_prev.reshape(batch_size, -1)
            mem_fc1 = self.leak_mem * mem_fc1 + self.bntt_fc[t](self.fc1(out_prev))
            mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0
            out = self.spike_fn(mem_thr)

            rst = torch.zeros_like(mem_fc1)
            rst[mem_thr > 0] = self.fc1.threshold
            mem_fc1 = mem_fc1 - rst
            out_prev = out

            mem_fc2 = mem_fc2 + self.fc2(out_prev)
            if return_time_outputs:
                time_outputs.append(mem_fc2 / float(t + 1))

        out_voltage = mem_fc2 / float(self.timesteps)
        if return_time_outputs:
            return out_voltage, torch.stack(time_outputs, dim=1)
        return out_voltage

    def forward_with_time_outputs(self, inp: torch.Tensor):
        return self.forward(inp, return_time_outputs=True)
