import copy
import os
import sys
import time
import logging
import shutil
import argparse
import torch
from tensorboardX import SummaryWriter
from typing import Callable
from functools import partial
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms
# from mmcv import Config
# from mmcv.cnn.utils import get_model_complexity_info
# from mmcv.cnn.utils.flops_counter import flops_to_string, params_to_string

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Unsupported value encountered.')


def run_pre(cfg):
	# from time
	if cfg.sleep > -1:
		for i in range(cfg.sleep):
			time.sleep(1)
			print('\rCount down : {} s'.format(cfg.sleep - 1 - i), end='')
	# from memory
	elif cfg.memory > -1:
		s_times = 0
		while True:
			os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Used > tmp')
			memory_used = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
			if memory_used[0] < 3000:
				os.system('rm tmp')
				break
			else:
				s_times += 1
				time.sleep(1)
				print('\rWaiting for {} s'.format(s_times), end='')


def makedirs(dirs, exist_ok=False):
	if not isinstance(dirs, list):
		dirs = [dirs]
	for dir in dirs:
		os.makedirs(dir, exist_ok=exist_ok)
	
	
def init_checkpoint(cfg):

	def rm_zero_size_file(path):
			files = os.listdir(path)
			for file in files:
				path = '{}/{}'.format(cfg.logdir, file)
				size = os.path.getsize(path)  # unit:B
				if os.path.isfile(path) and size < 8:
					os.remove(path)

	os.makedirs(cfg.trainer.checkpoint, exist_ok=True)
	if cfg.trainer.resume_dir:
		cfg.logdir = '{}/{}'.format(cfg.trainer.checkpoint, cfg.trainer.resume_dir)
		checkpoint_path = cfg.model.model_kwargs['checkpoint_path']
		if checkpoint_path == '':
			cfg.model.model_kwargs['checkpoint_path'] = '{}/latest_ckpt.pth'.format(cfg.logdir)
		else:
			cfg.model.model_kwargs['checkpoint_path'] = '{}/{}'.format(cfg.logdir, checkpoint_path.split('/')[-1])
		state_dict = torch.load(cfg.model.model_kwargs['checkpoint_path'], map_location='cpu')
		cfg.trainer.iter, cfg.trainer.epoch = state_dict['iter'], state_dict['epoch']
		cfg.trainer.topk_recorder = state_dict['topk_recorder']
	else:
		if cfg.master:
			logdir = '{}_{}_{}_{}'.format(cfg.trainer.name, cfg.model.name, cfg.data.type, time.strftime("%Y%m%d-%H%M%S"))
			cfg.logdir = '{}/{}'.format(cfg.trainer.checkpoint, logdir)
			os.makedirs(cfg.logdir, exist_ok=True)
			shutil.copy('{}.py'.format('/'.join(cfg.cfg_path.split('.'))), '{}/{}.py'.format(cfg.logdir, cfg.cfg_path.split('.')[-1]))
		else:
			cfg.logdir = None
		cfg.trainer.iter, cfg.trainer.epoch = 0, 0
		cfg.trainer.topk_recorder = dict()
		cfg.trainer.topk_recorder = dict(net_top1=[], net_top5=[], net_E_top1=[], net_E_top5=[])
	cfg.logger = get_logger(cfg) if cfg.master else None
	cfg.writer = SummaryWriter(log_dir=cfg.logdir, comment='') if cfg.master else None
	log_msg(cfg.logger, f'==> Logging on master GPU: {cfg.logger_rank}')
	# rm_zero_size_file(cfg.logdir) if cfg.master else None


def log_cfg(cfg):
	
	def _parse_Namespace(cfg, base_str=''):
		ret = {}
		if hasattr(cfg, '__dict__'):
			for key, val in cfg.__dict__.items():
				if not key.startswith('_'):
					ret.update(_parse_Namespace(val, '{}.{}'.format(base_str, key).lstrip('.')))
		else:
			ret.update({base_str:cfg})
		return ret
	
	cfg_dict = _parse_Namespace(cfg)
	key_max_length = max(list(map(len, cfg_dict.keys())))
	excludes = ['writer.', 'logger.handlers']
	exclude_keys = []
	for k, v in cfg_dict.items():
		for exclude in excludes:
			if k.find(exclude) != -1:
				exclude_keys.append(k) if k not in exclude_keys else None
	# cfg_str = '\n'.join(
	# 	[(('{' + ':<{}'.format(key_max_length) + '} : {' + ':<{}'.format(key_max_length)) + '}').format(k, str(v)) for
	# 	 k, v in cfg_dict.items()])
	cfg_str = ''
	for k, v in cfg_dict.items():
		if k in exclude_keys:
			continue
		cfg_str += ('{' + ':<{}'.format(key_max_length) + '} : {' + ':<{}'.format(key_max_length) + '}').format(k, str(v))
		cfg_str += '\n'
	cfg_str = cfg_str.strip()
	cfg.cfg_dict, cfg.cfg_str = cfg_dict, cfg_str
	log_msg(cfg.logger, f'==> ********** cfg ********** \n{cfg.cfg_str}')


def get_logger(cfg, mode='a+'):
	log_format = '%(asctime)s - %(message)s'
	logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
	fh = logging.FileHandler('{}/log_{}.txt'.format(cfg.logdir, cfg.mode), mode=mode)
	fh.setFormatter(logging.Formatter(log_format))
	logger = logging.getLogger()
	logger.addHandler(fh)
	cfg.logger = logger
	return logger

def able(ret, mark=False, default=None):
	return ret if mark else default


def log_msg(logger, msg, level='info'):
	if logger is not None:
		if msg is not None and level == 'info':
			logger.info(msg)


class AvgMeter(object):
	def __init__(self, name, fmt=':f', show_name='val', add_name=''):
		self.name = name
		self.fmt = fmt
		self.show_name = show_name
		self.add_name = add_name
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __str__(self):
		fmtstr = '[{name} {' + self.show_name + self.fmt + '}'
		fmtstr += (' ({' + self.add_name + self.fmt + '})]' if self.add_name else ']')
		return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
	def __init__(self, meters, default_prefix=""):
		self.iter_fmtstr_iter = '{}: {:>3.2f}% [{}/{}]'
		self.iter_fmtstr_batch = ' [{:<.1f}/{:<3.1f}]'
		self.meters = meters
		self.default_prefix = default_prefix

	def get_msg(self, iter, iter_full, epoch=None, epoch_full=None, prefix=None):
		entries = [self.iter_fmtstr_iter.format(prefix if prefix else self.default_prefix, iter / iter_full * 100, iter, iter_full, epoch, epoch_full)]
		if epoch:
			entries += [self.iter_fmtstr_batch.format(epoch, epoch_full)]
		for meter in self.meters.values():
			entries.append(str(meter)) if meter.count > 0 else None
		return ' '.join(entries)


def get_log_terms(log_terms, default_prefix=''):
	terms = {}
	for t in log_terms:
		t = {k: v for k, v in t.items()}
		t_name = t['name']
		terms[t_name] = AvgMeter(**t)
	progress = ProgressMeter(terms, default_prefix=default_prefix)
	return terms, progress


def update_log_term(term, val, n, master):
	term.update(val, n) if term and master else None


def accuracy(output, target, topk=(1,)):
	maxk = max(topk)
	batch_size = target.size(0)
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.reshape(1, -1).expand_as(pred))
	return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk], [correct[:k].reshape(-1).float().sum(0) for k in topk] + [batch_size]


def get_timepc():
	if torch.cuda.is_available():
		torch.cuda.synchronize()
	return time.perf_counter()


def get_net_params(net):
	num_params = 0
	for param in net.parameters():
		if param.requires_grad:
			num_params += param.numel()
	return num_params / 1e6


def import_abspy(name="models", path="classification/"):
	import sys
	import importlib
	path = os.path.abspath(path)
	assert os.path.isdir(path)
	sys.path.insert(0, path)
	module = importlib.import_module(name)
	sys.path.pop(0)
	return module


# used for print flops
class FLOPs:
	@staticmethod
	def register_supported_ops():
		build = import_abspy("lib_mamba", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../model"))
		selective_scan_flop_jit: Callable = build.vmamba.selective_scan_flop_jit
		flops_selective_scan_fn: Callable = build.csms6s.flops_selective_scan_fn
		flops_selective_scan_ref: Callable = build.csms6s.flops_selective_scan_ref

		supported_ops = {
			"aten::gelu": None,  # as relu is in _IGNORED_OPS
			"aten::silu": None,  # as relu is in _IGNORED_OPS
			"aten::neg": None,  # as relu is in _IGNORED_OPS
			"aten::exp": None,  # as relu is in _IGNORED_OPS
			"aten::flip": None,  # as permute is in _IGNORED_OPS
			# "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit,  # latter
			# "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,  # latter
			# "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,  # latter
			# "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,  # latter
			# "prim::PythonOp.SelectiveScan": selective_scan_flop_jit,  # latter
			# "prim::PythonOp.CrossScanTritonF": selective_scan_flop_jit,  # latter
			"prim::PythonOp.SelectiveScanCuda": partial(selective_scan_flop_jit, backend="prefixsum", verbose=False),
			# "prim::PythonOp.CrossMergeTritonF": selective_scan_flop_jit,  # latter
			# "aten::scaled_dot_product_attention": ...
		}
		return supported_ops

	@staticmethod
	def check_operations(model: torch.nn.Module, inputs=None, input_shape=(3, 224, 224)):
		from fvcore.nn.jit_analysis import _get_scoped_trace_graph, _named_modules_with_dup, Counter, JitModelAnalysis

		if inputs is None:
			assert input_shape is not None
			if len(input_shape) == 1:
				input_shape = (1, 3, input_shape[0], input_shape[0])
			elif len(input_shape) == 2:
				input_shape = (1, 3, *input_shape)
			elif len(input_shape) == 3:
				input_shape = (1, *input_shape)
			else:
				assert len(input_shape) == 4

			inputs = (torch.randn(input_shape).to(next(model.parameters()).device),)

		model.eval()

		flop_counter = JitModelAnalysis(model, inputs)
		flop_counter._ignored_ops = set()
		flop_counter._op_handles = dict()
		assert flop_counter.total() == 0  # make sure no operations supported
		print(flop_counter.unsupported_ops(), flush=True)
		print(f"supported ops {flop_counter._op_handles}; ignore ops {flop_counter._ignored_ops};", flush=True)

	@classmethod
	def fvcore_flop_count(cls, model: torch.nn.Module, inputs=None, input_shape=(3, 224, 224), show_table=True,
						  show_arch=False, verbose=True):
		supported_ops = cls.register_supported_ops()
		from fvcore.nn.parameter_count import parameter_count as fvcore_parameter_count
		from fvcore.nn.flop_count import flop_count, FlopCountAnalysis, _DEFAULT_SUPPORTED_OPS
		from fvcore.nn.print_model_statistics import flop_count_str, flop_count_table
		from fvcore.nn.jit_analysis import _IGNORED_OPS
		from fvcore.nn.jit_handles import get_shape, addmm_flop_jit

		if inputs is None:
			assert input_shape is not None
			if len(input_shape) == 1:
				input_shape = (1, 3, input_shape[0], input_shape[0])
			elif len(input_shape) == 2:
				input_shape = (1, 3, *input_shape)
			elif len(input_shape) == 3:
				input_shape = (1, *input_shape)
			else:
				assert len(input_shape) == 4

			inputs = (torch.randn(input_shape).to(next(model.parameters()).device),)

		model.eval()

		Gflops, unsupported = flop_count(model=model, inputs=inputs, supported_ops=supported_ops)

		flops_table = flop_count_table(
			flops=FlopCountAnalysis(model, inputs).set_op_handle(**supported_ops),
			max_depth=100,
			activations=None,
			show_param_shapes=True,
		)

		flops_str = flop_count_str(
			flops=FlopCountAnalysis(model, inputs).set_op_handle(**supported_ops),
			activations=None,
		)

		if show_arch:
			print(flops_str)

		if show_table:
			print(flops_table)

		params = fvcore_parameter_count(model)[""]
		flops = sum(Gflops.values())

		if verbose:
			print(Gflops.items())
			print("[GFlops: {:>6.3f}G]" "[Params: {:>6.3f}M]".format(flops, params / 1e6), flush=True)

		return params, flops

def get_val_dataloader(batch_size=64, root="./val", img_size=224, sequential=True):
	import torch.utils.data
	size = int((224 / 224) * img_size)
	transform = transforms.Compose([
		transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
		transforms.CenterCrop((img_size, img_size)),
		transforms.ToTensor(),
		transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
	])

	dataset = datasets.ImageFolder(root, transform=transform)
	if sequential:
		sampler = torch.utils.data.SequentialSampler(dataset)
	else:
		sampler = torch.utils.data.DistributedSampler(dataset)

	data_loader = torch.utils.data.DataLoader(
		dataset, sampler=sampler,
		batch_size=batch_size,
		shuffle=False,
		num_workers=0,
		pin_memory=True,
		drop_last=False
	)
	return data_loader

class Throughput:
    # default no amp in testing tp
    # copied from swin_transformer
    @staticmethod
    @torch.no_grad()
    def throughput(data_loader, model, logger=logging):
        model.eval()
        for idx, (images, _) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            batch_size = images.shape[0]
            for i in range(50):
                model(images)
            torch.cuda.synchronize()
            logger.info(f"throughput averaged with 30 times")
            torch.cuda.reset_peak_memory_stats()
            tic1 = time.time()
            for i in range(30):
                model(images)
            torch.cuda.synchronize()
            tic2 = time.time()
            logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
            logger.info(f"batch_size {batch_size} mem cost {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
            return

    @staticmethod
    @torch.no_grad()
    def throughputamp(data_loader, model, logger=logging):
        model.eval()

        for idx, (images, _) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            batch_size = images.shape[0]
            for i in range(50):
                with torch.cuda.amp.autocast():
                    model(images)
            torch.cuda.synchronize()
            logger.info(f"throughput averaged with 30 times")
            torch.cuda.reset_peak_memory_stats()
            tic1 = time.time()
            for i in range(30):
                with torch.cuda.amp.autocast():
                    model(images)
            torch.cuda.synchronize()
            tic2 = time.time()
            logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
            logger.info(f"batch_size {batch_size} mem cost {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
            return

    @staticmethod
    def testfwdbwd(data_loader, model, logger, amp=True):
        model.cuda().train()
        criterion = torch.nn.CrossEntropyLoss()

        for idx, (images, targets) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            batch_size = images.shape[0]
            for i in range(50):
                with torch.cuda.amp.autocast(enabled=amp):
                    out = model(images)
                    loss = criterion(out, targets)
                    loss.backward()
            torch.cuda.synchronize()
            logger.info(f"testfwdbwd averaged with 30 times")
            torch.cuda.reset_peak_memory_stats()
            tic1 = time.time()
            for i in range(30):
                with torch.cuda.amp.autocast(enabled=amp):
                    out = model(images)
                    loss = criterion(out, targets)
                    loss.backward()
            torch.cuda.synchronize()
            tic2 = time.time()
            logger.info(f"batch_size {batch_size} testfwdbwd {30 * batch_size / (tic2 - tic1)}")
            logger.info(f"batch_size {batch_size} mem cost {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
            return

    @classmethod
    def testall(cls, model, dataloader=None, data_path="", img_size=224, _batch_size=128, with_flops=False, inference_only=False):
        from fvcore.nn import parameter_count
        torch.cuda.empty_cache()
        model.cuda().eval()
        if with_flops:
            try:
                FLOPs.fvcore_flop_count(model, input_shape=(3, img_size, img_size), show_arch=False)
            except Exception as e:
                print("ERROR:", e, flush=True)
        # print(parameter_count(model)[""], sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
        if dataloader is None:
            dataloader = get_val_dataloader(
                batch_size=_batch_size,
                root=os.path.join(os.path.abspath(data_path), "val"),
                img_size=img_size,
            )
        print('begin')
        cls.throughput(data_loader=dataloader, model=model, logger=logging)
        print("finished")
        if inference_only:
            return
        PASS = False
        batch_size = _batch_size
        while (not PASS) and (batch_size > 0):
            try:
                _dataloader = get_val_dataloader(
                    batch_size=batch_size,
                    root=os.path.join(os.path.abspath(data_path), "val"),
                    img_size=img_size,
                )
                cls.testfwdbwd(data_loader=_dataloader, model=model, logger=logging)
                cls.testfwdbwd(data_loader=_dataloader, model=model, logger=logging, amp=False)
                PASS = True
            except Exception as e:
                print(e)
                batch_size = batch_size // 2
                print(f"batch_size {batch_size}", flush=True)


# TIME_MIX_EXTRA_DIM = 32
# TIME_DECAY_EXTRA_DIM = 64
#
# def vrwkv_flops(n, dim):
#     return n * dim * 29
#
# def vrwkv6_flops(n, dim, head_size):
#     addi_flops = 0
#     addi_flops += n * dim * (TIME_MIX_EXTRA_DIM * 10 + TIME_DECAY_EXTRA_DIM * 2 + 7 * head_size + 17)
#     addi_flops += n * (TIME_MIX_EXTRA_DIM * 5 + TIME_DECAY_EXTRA_DIM)
#     return addi_flops
#
# def get_addi_flops_vrwkv6(model, input_shape, cfg):
# 	_, H, W = input_shape
# 	try:
# 		patch_size = cfg.model.backbone.patch_size
# 	except:
# 		patch_size = 16
# 	h, w = H / patch_size, W / patch_size
#
# 	model_name = type(model.backbone).__name__
# 	embed_dims = model.backbone.embed_dims
# 	head_size = embed_dims // cfg.model.backbone.num_heads
# 	print(f"Head Size in VRWKV6: {head_size}")
# 	num_layers = len(model.backbone.layers)
# 	addi_flops = 0
# 	addi_flops += (num_layers * vrwkv6_flops(h * w, embed_dims, head_size))
# 	print(f"Additional Flops in VRWKV6*{num_layers} layers: {flops_to_string(addi_flops)}")
# 	return addi_flops
#
# def get_addi_flops_vrwkv(model, input_shape, cfg):
# 	_, H, W = input_shape
# 	try:
# 		patch_size = cfg.model.backbone.patch_size
# 	except:
# 		patch_size = 16
# 	h, w = H / patch_size, W / patch_size
#
# 	model_name = type(model.backbone).__name__
# 	embed_dims = model.backbone.embed_dims
# 	num_layers = len(model.backbone.layers)
# 	addi_flops = 0
# 	addi_flops += (num_layers * vrwkv_flops(h * w, embed_dims))
# 	print(f"Additional Flops in VRWKV(Attn)*{num_layers} layers: {flops_to_string(addi_flops)}")
# 	return addi_flops
#
# def get_flops(model, input_shape, cfg, ost):
# 	flops, params = get_model_complexity_info(model, input_shape, as_strings=False, ost=ost)
# 	model_name = type(model.backbone).__name__
# 	if model_name == 'VRWKV':
# 		add_flops = get_addi_flops_vrwkv(model, input_shape, cfg)
# 		flops += add_flops
# 	elif model_name == 'VRWKV6':
# 		add_flops = get_addi_flops_vrwkv6(model, input_shape, cfg)
# 		flops += add_flops
# 	return flops_to_string(flops), params_to_string(params)

	# equals with fvcore_flop_count
	# @classmethod
	# def mmengine_flop_count(cls, model: nn.Module = None, input_shape=(3, 224, 224), show_table=False, show_arch=False,
	# 						_get_model_complexity_info=False):
	# 	supported_ops = cls.register_supported_ops()
	# 	from mmengine.analysis.print_helper import is_tuple_of, FlopAnalyzer, ActivationAnalyzer, parameter_count, \
	# 		_format_size, complexity_stats_table, complexity_stats_str
	# 	from mmengine.analysis.jit_analysis import _IGNORED_OPS
	# 	from mmengine.analysis.complexity_analysis import _DEFAULT_SUPPORTED_FLOP_OPS, _DEFAULT_SUPPORTED_ACT_OPS
	# 	from mmengine.analysis import get_model_complexity_info as mm_get_model_complexity_info
	#
	# 	# modified from mmengine.analysis
	# 	def get_model_complexity_info(
	# 			model: nn.Module,
	# 			input_shape: Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...],
	# 			None] = None,
	# 			inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[Any, ...],
	# 			None] = None,
	# 			show_table: bool = True,
	# 			show_arch: bool = True,
	# 	):
	# 		if input_shape is None and inputs is None:
	# 			raise ValueError('One of "input_shape" and "inputs" should be set.')
	# 		elif input_shape is not None and inputs is not None:
	# 			raise ValueError('"input_shape" and "inputs" cannot be both set.')
	#
	# 		if inputs is None:
	# 			device = next(model.parameters()).device
	# 			if is_tuple_of(input_shape, int):  # tuple of int, construct one tensor
	# 				inputs = (torch.randn(1, *input_shape).to(device),)
	# 			elif is_tuple_of(input_shape, tuple) and all([
	# 				is_tuple_of(one_input_shape, int)
	# 				for one_input_shape in input_shape  # type: ignore
	# 			]):  # tuple of tuple of int, construct multiple tensors
	# 				inputs = tuple([
	# 					torch.randn(1, *one_input_shape).to(device)
	# 					for one_input_shape in input_shape  # type: ignore
	# 				])
	# 			else:
	# 				raise ValueError(
	# 					'"input_shape" should be either a `tuple of int` (to construct'
	# 					'one input tensor) or a `tuple of tuple of int` (to construct'
	# 					'multiple input tensors).')
	#
	# 		flop_handler = FlopAnalyzer(model, inputs).set_op_handle(**supported_ops)
	# 		# activation_handler = ActivationAnalyzer(model, inputs)
	#
	# 		flops = flop_handler.total()
	# 		# activations = activation_handler.total()
	# 		params = parameter_count(model)['']
	#
	# 		flops_str = _format_size(flops)
	# 		# activations_str = _format_size(activations)
	# 		params_str = _format_size(params)
	#
	# 		if show_table:
	# 			complexity_table = complexity_stats_table(
	# 				flops=flop_handler,
	# 				# activations=activation_handler,
	# 				show_param_shapes=True,
	# 			)
	# 			complexity_table = '\n' + complexity_table
	# 		else:
	# 			complexity_table = ''
	#
	# 		if show_arch:
	# 			complexity_arch = complexity_stats_str(
	# 				flops=flop_handler,
	# 				# activations=activation_handler,
	# 			)
	# 			complexity_arch = '\n' + complexity_arch
	# 		else:
	# 			complexity_arch = ''
	#
	# 		return {
	# 			'flops': flops,
	# 			'flops_str': flops_str,
	# 			# 'activations': activations,
	# 			# 'activations_str': activations_str,
	# 			'params': params,
	# 			'params_str': params_str,
	# 			'out_table': complexity_table,
	# 			'out_arch': complexity_arch
	# 		}
	#
	# 	if _get_model_complexity_info:
	# 		return get_model_complexity_info
	#
	# 	model.eval()
	# 	analysis_results = get_model_complexity_info(
	# 		model,
	# 		input_shape,
	# 		show_table=show_table,
	# 		show_arch=show_arch,
	# 	)
	# 	flops = analysis_results['flops_str']
	# 	params = analysis_results['params_str']
	# 	# activations = analysis_results['activations_str']
	# 	out_table = analysis_results['out_table']
	# 	out_arch = analysis_results['out_arch']
	#
	# 	if show_arch:
	# 		print(out_arch)
	#
	# 	if show_table:
	# 		print(out_table)
	#
	# 	split_line = '=' * 30
	# 	print(f'{split_line}\nInput shape: {input_shape}\t'
	# 		  f'Flops: {flops}\tParams: {params}\t'
	# 		  #   f'Activation: {activations}\n{split_line}'
	# 		  , flush=True)
	#
	# # print('!!!Only the backbone network is counted in FLOPs analysis.')
	# # print('!!!Please be cautious if you use the results in papers. '
	# #       'You may need to check if all ops are supported and verify that the '
	# #       'flops computation is correct.')
	#
	# @classmethod
	# def mmdet_flops(cls, config=None, extra_config=None):
	# 	from mmengine.config import Config
	# 	from mmengine.runner import Runner
	# 	import numpy as np
	# 	import os
	#
	# 	cfg = Config.fromfile(config)
	# 	if "model" in cfg:
	# 		if "pretrained" in cfg["model"]:
	# 			cfg["model"].pop("pretrained")
	# 	if extra_config is not None:
	# 		new_cfg = Config.fromfile(extra_config)
	# 		new_cfg["model"] = cfg["model"]
	# 		cfg = new_cfg
	# 	cfg["work_dir"] = "/tmp"
	# 	cfg["default_scope"] = "mmdet"
	# 	runner = Runner.from_cfg(cfg)
	# 	model = runner.model.cuda()
	# 	get_model_complexity_info = cls.mmengine_flop_count(_get_model_complexity_info=True)
	#
	# 	if True:
	# 		oridir = os.getcwd()
	# 		os.chdir(os.path.join(os.path.dirname(__file__), "../detection"))
	# 		data_loader = runner.val_dataloader
	# 		num_images = 100
	# 		mean_flops = []
	# 		for idx, data_batch in enumerate(data_loader):
	# 			if idx == num_images:
	# 				break
	# 			data = model.data_preprocessor(data_batch)
	# 			model.forward = partial(model.forward, data_samples=data['data_samples'])
	# 			# out = get_model_complexity_info(model, inputs=data['inputs'])
	# 			out = get_model_complexity_info(model, input_shape=(3, 1280, 800))
	# 			params = out['params_str']
	# 			mean_flops.append(out['flops'])
	# 		mean_flops = np.average(np.array(mean_flops))
	# 		print(params, mean_flops)
	# 		os.chdir(oridir)
	#
	# @classmethod
	# def mmseg_flops(cls, config=None, input_shape=(3, 512, 2048)):
	# 	from mmengine.config import Config
	# 	from mmengine.runner import Runner
	#
	# 	cfg = Config.fromfile(config)
	# 	cfg["work_dir"] = "/tmp"
	# 	cfg["default_scope"] = "mmseg"
	# 	runner = Runner.from_cfg(cfg)
	# 	model = runner.model.cuda()
	#
	# 	cls.fvcore_flop_count(model, input_shape=input_shape)
