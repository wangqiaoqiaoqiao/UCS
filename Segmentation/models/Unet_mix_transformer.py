from Segmentation.models.decode_heads.segformer_head import SegFormerHead
from Segmentation.models.backbones.mix_transformer import *
import torch.nn as nn

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    print('Warning',
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class Unet_mit(nn.Module):
    def __init__(self, encoder_name, inchannels, decoder_key, pretrained,align_corners =False):
        super(Unet_mit, self).__init__()
        self.align_corners = align_corners
        if encoder_name == 'mit_b0':
            self.encoder = mit_b0()
        elif encoder_name == 'mit_b1':
            self.encoder = mit_b1(inchannels=inchannels)
        elif encoder_name == 'mit_b2':
            self.encoder = mit_b2()
        elif encoder_name == 'mit_b3':
            self.encoder = mit_b3()
        self.decoder = SegFormerHead(feature_strides=decoder_key['feature_strides'],
                                     in_channels=decoder_key['in_channels'],
                                     in_index=decoder_key['in_index'],
                                     channels=decoder_key['channels'],
                                     dropout_ratio=decoder_key['dropout_ratio'],
                                     num_classes=decoder_key['num_classes'],
                                     norm_cfg=decoder_key['norm_cfg'],
                                     align_corners=decoder_key['align_corners'],
                                     decoder_params=decoder_key['decoder_params'],
                                     loss_decode=decoder_key['loss_decode']
                                     )

        self.init_checkpoint(pretrained)
    def init_checkpoint(self,pretrained):
        checkpoint = torch.load(pretrained)
        checkpoint.popitem()
        checkpoint.popitem()
        if self.encoder.state_dict()['patch_embed1.proj.weight'].shape == torch.Size([64, 1, 7, 7]):
            checkpoint.pop('patch_embed1.proj.weight')
        self.encoder.load_state_dict(checkpoint, strict=False)
    def forward(self,x):
        origin_x = x
        x = self.encoder(x)
        output = self.decoder(x)

        output = resize(
                input=output,
                size=origin_x.shape[-1],
                mode='bilinear',
                align_corners=self.align_corners)

        return output