import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Sep_STS_Encoder import ResBlock
import cupy_module.adacof as adacof
def joinTensors(X1 , X2 , type="concat"):
    if type == "add":
        return X1 + X2
    elif type == "concat":
        return torch.cat([X1 , X2] , dim=1)
    else:
        return X1
    
class upSplit(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.upconv = nn.ModuleList(
                [nn.ConvTranspose3d(out_channels=out_ch,in_channels=in_ch, kernel_size=(3,3,3), stride=(1,2,2), padding=1),
                 ]
            )
        self.upconv = nn.Sequential(*self.upconv)

    def forward(self, x, output_size):
        x = self.upconv[0](x, output_size=output_size)
        return x

class Conv_2d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, padding=0, bias=False, batchnorm=False, stride=1):
        super().__init__()
        self.conv = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=bias,stride=stride)]
        if batchnorm:
            self.conv += [nn.BatchNorm2d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        return self.conv(x)



class Conv_3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, use_batchnorm=False):

        super().__init__()
        self.conv_layers = [nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
]

        if use_batchnorm:
            self.conv_layers += [nn.BatchNorm3d(out_channels)]

        self.conv = nn.Sequential(*self.conv_layers)

    def forward(self, x):  
        return self.conv(x)




    def forward(self, input_frames):
        input_images = torch.stack(input_frames, dim=2)
        _, _, _, height, width = input_images.shape

        mean_value = input_images.mean(2, keepdim=True).mean(3, keepdim=True).mean(4, keepdim=True)
        input_images = input_images - mean_value

        encoded_0, encoded_1, encoded_2, encoded_3, encoded_4 = self.encoder(input_images)

        decoded_3 = self.lrelu(self.decoder[0](encoded_4, encoded_3.size()))
        decoded_3 = joinTensors(decoded_3 , encoded_3 , type=self.joinType)

        decoded_2 = self.lrelu(self.decoder[1](decoded_3, encoded_2.size()))
        decoded_2 = joinTensors(decoded_2 , encoded_2 , type=self.joinType)

        decoded_1 = self.lrelu(self.decoder[2](decoded_2, encoded_1.size()))
        decoded_1 = joinTensors(decoded_1 , encoded_1 , type=self.joinType)

        feature_3 = self.smooth_ll(decoded_3)
        feature_2 = self.smooth_l(decoded_2)
        feature_1 = self.smooth(decoded_1)

        output_ll = self.predict_ll(feature_3, input_frames, encoded_2.size()[-2:])

        output_l = self.predict_l(feature_2, input_frames, encoded_1.size()[-2:])
        output_l = F.interpolate(output_ll, size=output_l.size()[-2:], mode='bilinear') + output_l

        final_output = self.predict(feature_1, input_frames, encoded_0.size()[-2:])
        final_output = F.interpolate(output_l, size=final_output.size()[-2:], mode='bilinear') + final_output

        if self.training:
            return output_ll, output_l, final_output
        else:
            return final_output

            # return out_ll, out_l, out
class UNet_3D_3D(nn.Module):
    def __init__(self, num_inputs=4, merge_type="concatenate", kernel_size=5, dilation_rate=1):
        super().__init__()

        num_filters = [512, 256, 128, 64]
        window_sizes = [(1, 8, 8), (1, 8, 8), (1, 8, 8), (1, 8, 8)]
        num_heads = [2, 4, 8, 16]
        self.merge_type = merge_type
        self.num_inputs = num_inputs

        growth_rate = 2 if merge_type == "concatenate" else 1
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        from model.Sep_STS_Encoder import SepSTSEncoder as CustomEncoder
        self.encoder = CustomEncoder(num_filters, num_inputs, window_size=window_sizes, nh=num_heads)

        self.decoder = nn.Sequential(
            upSplit(num_filters[0], num_filters[1]),
            upSplit(num_filters[1]*growth_rate, num_filters[2]),
            upSplit(num_filters[2]*growth_rate, num_filters[3]),
        )

        def SmoothNetwork(input_channels, output_channels):
            return torch.nn.Sequential(
                Conv_3d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, use_batchnorm=False),
                ResBlock(output_channels, kernel_size=3),
            )

        output_channels = 64
        self.smooth_ll =SmoothNetwork(num_filters[1]*growth_rate, output_channels)
        self.smooth_l = SmoothNetwork(num_filters[2]*growth_rate, output_channels)
        self.smooth = SmoothNetwork(num_filters[3]*growth_rate, output_channels)

        self.predict_ll = SynBlock(num_inputs, output_channels, ks=kernel_size, dilation=dilation_rate, normalize_weight=True)
        self.predict_l = SynBlock(num_inputs, output_channels, ks=kernel_size, dilation=dilation_rate, normalize_weight=False)
        self.predict = SynBlock(num_inputs, output_channels, ks=kernel_size, dilation=dilation_rate, normalize_weight=False)

class MySequential(nn.Sequential):
    def forward(self, input, output_size):
        for module in self:
            if isinstance(module, nn.ConvTranspose2d):
                input = module(input, output_size)
            else:
                input = module(input)
        return input


class SynBlock(nn.Module):
    def __init__(self, num_inputs, num_features, kernel_size, dilation, normalize_weight=True):
        super(SynBlock, self).__init__()

        def SubnetOffset(kernel_size):
            return MySequential(
                torch.nn.Conv2d(in_channels=num_features, out_channels=num_features, stride=1, kernel_size=3, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Conv2d(kernel_size=3, stride=1, padding=1, in_channels=num_features, out_channels=kernel_size),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.ConvTranspose2d(kernel_size, kernel_size, stride=2, padding=1, kernel_size=3),
                torch.nn.Conv2d(in_channels=kernel_size, out_channels=kernel_size, kernel_size=3, stride=1, padding=1)
            )

        def SubnetWeight(kernel_size):
            return MySequential(
                torch.nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Conv2d(in_channels=num_features, out_channels=kernel_size, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.ConvTranspose2d(kernel_size, kernel_size, kernel_size=3, stride=2, padding=1),
                torch.nn.Conv2d(in_channels=kernel_size, out_channels=kernel_size, kernel_size=3, stride=1, padding=1),
                nn.Softmax(dim=1) if normalize_weight else nn.Identity()
            )

        def SubnetOcclusion():
            return MySequential(
                torch.nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.ConvTranspose2d(num_features, num_features, kernel_size=3, stride=2, padding=1),
                torch.nn.Conv2d(in_channels=num_features, out_channels=num_inputs, kernel_size=3, stride=1, padding=1),
                torch.nn.Softmax(dim=1)
            )

        self.num_inputs = num_inputs
        self.kernel_size = kernel_size
        self.kernel_pad = int(((kernel_size - 1) * dilation) / 2.0)
        self.dilation = dilation

        self.pad_module = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])
       
        self.adacof_module = adacof.FunctionAdaCoF.apply

        self.weight_module = SubnetWeight(kernel_size ** 2)
        self.alpha_module = SubnetOffset(kernel_size ** 2)
        self.beta_module = SubnetOffset(kernel_size ** 2)
        self.occlusion_module = SubnetOcclusion()

        self.feature_fusion = Conv_2d(num_features * num_inputs, num_features, kernel_size=1, stride=1, use_batchnorm=False, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, feature, frames, output_size):
        height, width = output_size

        occlusion = torch.cat(torch.unbind(feature, 1), 1)
        occlusion = self.leaky_relu(self.feature_fusion(occlusion))
        occlusion_map = self.occlusion_module(occlusion, (height, width))

        batch_size, channels, time_steps, cur_height, cur_width = feature.shape
        feature = feature.transpose(1, 2).reshape(batch_size * time_steps, channels, cur_height, cur_width)
        betas = self.beta_module(feature, (height, width)).view(batch_size, time_steps, -1, height, width)
        alphas = self.alpha_module(feature, (height, width)).view(batch_size, time_steps, -1, height, width)
        weights = self.weight_module(feature, (height, width)).view(batch_size, time_steps, -1, height, width)
        

        warped_frames = []
        for i in range(self.num_inputs):
            weight = weights[:, i].contiguous()
            alpha = alphas[:, i].contiguous()
            beta = betas[:, i].contiguous()
            occlusion_map = occlusion_map[:, i:i+1]
            frame = F.interpolate(frames[i], size=weight.size()[-2:], mode='bilinear')

            warped_frames.append(
                occlusion_map * self.adacof_module(self.pad_module(frame), weight, alpha, beta, self.dilation)
            )

        reconstructed_frame = sum(warped_frames)
        return reconstructed_frame

    def __init__(self, n_inputs, nf, ks, dilation, norm_weight=True):
        super(SynBlock, self).__init__()

        def Subnet_offset(ks):
            return MySequential(
                torch.nn.Conv2d(in_channels=nf, out_channels=nf, stride=1, kernel_size=3, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Conv2d(kernel_size=3, stride=1, padding=1,in_channels=nf, out_channels=ks),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.ConvTranspose2d(ks, ks, stride=2, padding=1,kernel_size=3 ),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1)
            )

        def Subnet_weight(ks):
            return MySequential(
                torch.nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Conv2d(in_channels=nf, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.ConvTranspose2d(ks, ks, kernel_size=3, stride=2, padding=1),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1),
                nn.Softmax(1) if norm_weight else nn.Identity()
            )

        def Subnet_occlusion():
            return MySequential(
                torch.nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.ConvTranspose2d(nf, nf, kernel_size=3, stride=2, padding=1),
                torch.nn.Conv2d(in_channels=nf, out_channels=n_inputs, kernel_size=3, stride=1, padding=1),
                torch.nn.Softmax(dim=1)
            )

        self.n_inputs = n_inputs
        self.kernel_size = ks
        self.kernel_pad = int(((ks - 1) * dilation) / 2.0)
        self.dilation = dilation

        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])
       
        self.moduleAdaCoF = adacof.FunctionAdaCoF.apply

        self.ModuleWeight = Subnet_weight(ks ** 2)
        self.ModuleAlpha = Subnet_offset(ks ** 2)
        self.ModuleBeta = Subnet_offset(ks ** 2)
        self.moduleOcclusion = Subnet_occlusion()

        self.feature_fuse = Conv_2d(nf * n_inputs, nf, kernel_size=1, stride=1, batchnorm=False, bias=True)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, fea, frames, output_size):
        H, W = output_size

        occ = torch.cat(torch.unbind(fea, 1), 1)
        occ = self.lrelu(self.feature_fuse(occ))
        Occlusion = self.moduleOcclusion(occ, (H, W))

        B, C, T, cur_H, cur_W = fea.shape
        fea = fea.transpose(1, 2).reshape(B*T, C, cur_H, cur_W)
        betas = self.ModuleBeta(fea, (H, W)).view(B, T, -1, H, W)
        alphas = self.ModuleAlpha(fea, (H, W)).view(B, T, -1, H, W)
        weights = self.ModuleWeight(fea, (H, W)).view(B, T, -1, H, W)
        

        warp = []
        for i in range(self.n_inputs):
            weight = weights[:, i].contiguous()
            alpha = alphas[:, i].contiguous()
            beta = betas[:, i].contiguous()
            occ = Occlusion[:, i:i+1]
            frame = F.interpolate(frames[i], size=weight.size()[-2:], mode='bilinear')

            warp.append(
                occ * self.moduleAdaCoF(self.modulePad(frame), weight, alpha, beta, self.dilation)
            )

        framet = sum(warp)
        return framet

if __name__ == '__main__':
    model = UNet_3D_3D('unet_18', n_inputs=4, n_outputs=1)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('the total number of network parameters: {}'.format(total_params))
