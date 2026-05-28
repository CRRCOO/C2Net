import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.PVTv2 import pvt_v2_b2
from Model.MobileNetV2 import MobileNetV2
from Model.Modules import ConvBN, ConvBNGeLU
from Model.vmamba import SS2D, Mlp, selective_scan_fn_v1
from timm.models.layers import DropPath
import math
from einops import rearrange, repeat
from Model.CaLLE import CaLLE

import torchvision.transforms as T


class BasicConv(nn.Module):
	def __init__(self, inp, oup, stride=1, expand_ratio=4, dilation=(1, 2, 3), residual=True):
		super(BasicConv, self).__init__()
		self.stride = stride
		assert stride in [1, 2]

		hidden_dim = int(round(inp * expand_ratio))
		if self.stride == 1 and inp == oup:
			self.use_res_connect = residual
		else:
			self.use_res_connect = False

		self.conv1 = ConvBNGeLU(inp, hidden_dim, kernel_size=1)

		self.hidden_conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation[0], groups=hidden_dim,
		                              dilation=dilation[0])
		self.hidden_conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation[1], groups=hidden_dim,
		                              dilation=dilation[1])
		self.hidden_conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation[2], groups=hidden_dim,
		                              dilation=dilation[2])
		self.hidden_bnact = nn.Sequential(nn.BatchNorm2d(hidden_dim), nn.GELU())
		self.out_conv = nn.Sequential(
			nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
			nn.BatchNorm2d(oup),
		)

	def forward(self, x):
		m = self.conv1(x)
		m = self.hidden_conv1(m) + self.hidden_conv2(m) + self.hidden_conv3(m)
		m = self.hidden_bnact(m)
		if self.use_res_connect:
			return x + self.out_conv(m)
		else:
			return self.out_conv(m)


class DeBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(DeBlock, self).__init__()
		self.block = BasicConv(in_channels, out_channels)

	def forward(self, x):
		return self.block(x)


class CrossAttentionSSM(nn.Module):
	def __init__(
		self,
		# basic dims ===========
		d_model,
		d_state=4,
		ssm_ratio=1,
		dt_rank="auto",
		# dt init ==============
		dt_min=0.001,
		dt_max=0.1,
		dt_init="random",
		dt_scale=1.0,
		dt_init_floor=1e-4,
		# ======================
		**kwargs,
	):
		factory_kwargs = {"device": None, "dtype": None}
		super().__init__()
		self.d_model = d_model
		self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
		self.expand = ssm_ratio
		self.d_inner = int(self.expand * self.d_model)
		self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

		# x proj; dt proj ============================
		self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)

		self.dt_proj = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
		                            **factory_kwargs)

		# A, D =======================================
		self.A_log = self.A_log_init(self.d_state, self.d_inner)  # (D, N)
		self.D = self.D_init(self.d_inner)  # (D)

		# out norm ===================================
		self.out_norm = nn.LayerNorm(self.d_inner)

	@staticmethod
	def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
	            **factory_kwargs):
		dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

		# Initialize special dt projection to preserve variance at initialization
		dt_init_std = dt_rank ** -0.5 * dt_scale
		if dt_init == "constant":
			nn.init.constant_(dt_proj.weight, dt_init_std)
		elif dt_init == "random":
			nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
		else:
			raise NotImplementedError

		# Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
		dt = torch.exp(
			torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
			+ math.log(dt_min)
		).clamp(min=dt_init_floor)
		# Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
		inv_dt = dt + torch.log(-torch.expm1(-dt))
		with torch.no_grad():
			dt_proj.bias.copy_(inv_dt)
		# Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
		# dt_proj.bias._no_reinit = True

		return dt_proj

	@staticmethod
	def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
		# S4D real initialization
		A = repeat(
			torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
			"n -> d n",
			d=d_inner,
		).contiguous()
		A_log = torch.log(A)  # Keep A_log in fp32
		if copies > 0:
			A_log = repeat(A_log, "d n -> r d n", r=copies)
			if merge:
				A_log = A_log.flatten(0, 1)
		A_log = nn.Parameter(A_log)
		A_log._no_weight_decay = True
		return A_log

	@staticmethod
	def D_init(d_inner, copies=-1, device=None, merge=True):
		# D "skip" parameter
		D = torch.ones(d_inner, device=device)
		if copies > 0:
			D = repeat(D, "n1 -> r n1", r=copies)
			if merge:
				D = D.flatten(0, 1)
		D = nn.Parameter(D)  # Keep in fp32
		D._no_weight_decay = True
		return D

	def forward(self, f: torch.Tensor, x: torch.Tensor):
		"""
        f: fused multi-modality feature
        x: feature to be embbeded into f
        """
		selective_scan = selective_scan_fn_v1
		B, L, d = x.shape
		x = x.permute(0, 2, 1)
		x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
		dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
		dt = self.dt_proj.weight @ dt.t()
		dt = rearrange(dt, "d (b l) -> b d l", l=L)
		A = -torch.exp(self.A_log.float())  # (k * d, d_state)
		B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
		C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()

		f = f.permute(0, 2, 1)
		y = selective_scan(
			f, dt,
			A, B, C, self.D.float(),
			delta_bias=self.dt_proj.bias.float(),
			delta_softplus=True,
		)
		# assert out_y.dtype == torch.float
		y = rearrange(y, "b d l -> b l d")
		y = self.out_norm(y)
		return y


class SSM(nn.Module):
	def __init__(
		self,
		# (B, dim, dim, C)
		hidden_dim: int,
		# drop_path_rate
		drop_path: float = 0.2,
		attn_drop_rate: float = 0,
		d_state: int = 16,
		ssm_ratio=2.0,
		shared_ssm=False,
		softmax_version=False,
		use_checkpoint: bool = False,
		mlp_ratio=0.0,
		act_layer=nn.GELU,
		drop: float = 0.0,
		**kwargs,
	):
		super().__init__()
		self.use_checkpoint = use_checkpoint
		self.norm = nn.LayerNorm(normalized_shape=hidden_dim, eps=1e-6)
		# SS2DBlock in Fig. 8
		# SS2D: 2D-Selective-Scan
		self.op = SS2D(
			d_model=hidden_dim,
			dropout=attn_drop_rate,
			d_state=d_state,
			ssm_ratio=ssm_ratio,
			dt_rank="auto",
			shared_ssm=shared_ssm,
			softmax_version=softmax_version,
			**kwargs
		)
		self.drop_path = DropPath(drop_path)

		self.mlp_branch = mlp_ratio > 0
		if self.mlp_branch:
			self.norm2 = nn.LayerNorm(normalized_shape=hidden_dim, eps=1e-6)
			mlp_hidden_dim = int(hidden_dim * mlp_ratio)
			self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
			               channels_first=False)

	def forward(self, input: torch.Tensor):
		# input: (B, H, W, C)
		x = input + self.drop_path(self.op(self.norm(input)))
		if self.mlp_branch:
			x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
		return x


class CrossModalSSMFusion(nn.Module):
	"""CAMF Module"""
	def __init__(self, channel):
		super(CrossModalSSMFusion, self).__init__()

		self.ssm_x = SSM(hidden_dim=channel)
		self.ssm_dop = SSM(hidden_dim=channel)
		self.ssm_f = SSM(hidden_dim=channel)

		self.cross_ssm_xf = CrossAttentionSSM(d_model=channel)
		self.cross_ssm_dopf = CrossAttentionSSM(d_model=channel)

		self.conv = ConvBNGeLU(channel * 2, channel, kernel_size=3)

		self.mlp = nn.Sequential(
			nn.LayerNorm(normalized_shape=channel, eps=1e-6),
			nn.Linear(in_features=channel, out_features=channel // 2),
			nn.GELU(),
			nn.Linear(in_features=channel // 2, out_features=1)
		)

		self.f_conv = ConvBN(channel, channel, kernel_size=3)
		self.dop_conv = ConvBN(channel, channel, kernel_size=3)
		self.out_conv = ConvBN(channel, channel, kernel_size=3)

	def forward(self, x, y):
		B, C, H, W = x.shape

		# B,H,W,C
		xt = torch.permute(x, dims=(0, 2, 3, 1))
		yt = torch.permute(x, dims=(0, 2, 3, 1))

		# B,H,W,C
		xt = self.ssm_x(xt)
		yt = self.ssm_dop(yt)

		# B,H,W,C
		xy = torch.cat((x, y), dim=1)
		xy = self.conv(xy)
		xyt = torch.permute(xy, dims=(0, 2, 3, 1))
		xyt = self.ssm_f(xyt)

		# B,H,W,1
		w = self.mlp(xyt)
		w = torch.sigmoid(w)

		# B,HW,C
		xyt = xyt.reshape(B, H * W, C)
		xt = xt.reshape(B, H * W, C)
		yt = yt.reshape(B, H * W, C)
		x_f = self.cross_ssm_xf(f=xyt, x=xt)
		dop_f = self.cross_ssm_dopf(f=xyt, x=yt)

		# B,H,W,C
		x_f = x_f.reshape(B, H, W, C)
		dop_f = dop_f.reshape(B, H, W, C)
		x_f = x_f * w
		dop_f = dop_f * (1.0 - w)

		# B,C,H,W
		x_f = torch.permute(x_f, dims=(0, 3, 1, 2))
		dop_f = torch.permute(dop_f, dims=(0, 3, 1, 2))
		x_f = self.f_conv(x_f)
		dop_f = self.dop_conv(dop_f)
		f_f = torch.relu(x_f + dop_f)
		return self.out_conv(f_f)


class C2Net(nn.Module):

	def __init__(self, channels=48, lle_channels=16):
		super(C2Net, self).__init__()

		self.encoder = pvt_v2_b2()
		self.dop_encoder = MobileNetV2(in_channel=1)

		# natural lle fusion
		self.nlf = CaLLE(channel=lle_channels)
		# RGB normalization
		self.rgb_norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		# reduction
		vit_stage_channels = self.encoder.get_stage_channels()
		self.x_re_conv1 = ConvBNGeLU(in_channels=vit_stage_channels[0], out_channels=channels, kernel_size=3)
		self.x_re_conv2 = ConvBNGeLU(in_channels=vit_stage_channels[1], out_channels=channels, kernel_size=3)
		self.x_re_conv3 = ConvBNGeLU(in_channels=vit_stage_channels[2], out_channels=channels, kernel_size=3)
		self.x_re_conv4 = ConvBNGeLU(in_channels=vit_stage_channels[3], out_channels=channels, kernel_size=3)
		# reduction
		cnn_stage_channels = self.dop_encoder.get_stage_channels()
		self.y_re_conv1 = ConvBNGeLU(in_channels=cnn_stage_channels[1], out_channels=channels, kernel_size=3)
		self.y_re_conv2 = ConvBNGeLU(in_channels=cnn_stage_channels[2], out_channels=channels, kernel_size=3)
		self.y_re_conv3 = ConvBNGeLU(in_channels=cnn_stage_channels[3], out_channels=channels, kernel_size=3)
		self.y_re_conv4 = ConvBNGeLU(in_channels=cnn_stage_channels[4], out_channels=channels, kernel_size=3)
		# activation
		self.gelu = nn.GELU()
		# fusion
		self.fusion1 = CrossModalSSMFusion(channels)
		self.fusion2 = CrossModalSSMFusion(channels)
		self.fusion3 = CrossModalSSMFusion(channels)
		self.fusion4 = CrossModalSSMFusion(channels)
		# decoder:
		self.deconv3 = DeBlock(channels, channels)
		self.deconv2 = DeBlock(channels, channels)
		self.deconv1 = DeBlock(channels, channels)
		# out conv
		self.out_conv1 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
		self.out_conv2 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
		self.out_conv3 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
		self.out_conv4 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

	def forward(self, x, dop):
		# x: RGB  y: low-light enhanced RGB
		e_x, x_r, x_cod = self.nlf(x)
		x1, x2, x3, x4 = self.encoder(self.rgb_norm(e_x))
		_, y1, y2, y3, y4 = self.dop_encoder(dop)
		# channel reduction to 64
		x1 = self.x_re_conv1(x1)
		x2 = self.x_re_conv2(x2)
		x3 = self.x_re_conv3(x3)
		x4 = self.x_re_conv4(x4)
		y1 = self.y_re_conv1(y1)
		y2 = self.y_re_conv2(y2)
		y3 = self.y_re_conv3(y3)
		y4 = self.y_re_conv4(y4)
		# fusion
		f1 = self.fusion1(x1, y1)
		f2 = self.fusion2(x2, y2)
		f3 = self.fusion3(x3, y3)
		f4 = self.fusion4(x4, y4)
		# decoding
		out4 = f4
		out3 = self.gelu(
			self.deconv3(F.interpolate(out4, size=x3.shape[2:], mode='bilinear', align_corners=False)) + f3)
		out2 = self.gelu(
			self.deconv2(F.interpolate(out3, size=x2.shape[2:], mode='bilinear', align_corners=False)) + f2)
		out1 = self.gelu(
			self.deconv1(F.interpolate(out2, size=x1.shape[2:], mode='bilinear', align_corners=False)) + f1)

		out1 = self.out_conv1(out1)
		out2 = self.out_conv2(out2)
		out3 = self.out_conv3(out3)
		out4 = self.out_conv4(out4)

		# upsampling coarse maps and edge maps to gt size
		size = (out1.shape[2] * 4, out1.shape[3] * 4)
		out1 = F.interpolate(out1, size=size, mode='bilinear', align_corners=False)
		out2 = F.interpolate(out2, size=size, mode='bilinear', align_corners=False)
		out3 = F.interpolate(out3, size=size, mode='bilinear', align_corners=False)
		out4 = F.interpolate(out4, size=size, mode='bilinear', align_corners=False)

		return out1, out2, out3, out4, e_x, x_r, x_cod


if __name__ == '__main__':
	from utils.tools import get_model_complexity

	model = C2Net(channels=48, lle_channels=16).cuda()

	in_tensor = torch.randn(1, 3, 352, 352).cuda()
	dop_tensor = torch.randn(1, 1, 352, 352).cuda()
	macs, params = get_model_complexity(model, inputs=(in_tensor, dop_tensor), round=3)
	print(params, macs)
