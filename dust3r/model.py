# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
from copy import deepcopy
import ast
import math
import torch
import os
import re
from packaging import version
import huggingface_hub

from .utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from .heads import head_factory
from dust3r.patch_embed import get_patch_embed, ManyAR_PatchEmbed
from third_party.raft import load_RAFT

import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # noqa

inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), "Outdated huggingface_hub version, please reinstall requirements.txt"


_MAST3R_DROP_KWARGS = {
    "two_confs",
    "desc_conf_mode",
    "desc_mode",
    "desc_dim",
    "desc_norm",
    "desc_loss",
    "desc_loss_weight",
    "desc_proj_dim",
}


def _rewrite_mast3r_args_for_dust3r(args: str) -> tuple[str, bool]:
    """Adapt MASt3R checkpoint model args to this repo's DUSt3R model class.

    This repo does not ship AsymmetricMASt3R/descriptor heads. For geometry-only
    inference we map to AsymmetricCroCo3DStereo with a DPT points head.
    """
    if "AsymmetricMASt3R" not in args:
        return args, False

    # First try an AST-level rewrite for robust parsing across quote styles/spacing.
    try:
        expr = ast.parse(args, mode="eval")
        call = expr.body
        if isinstance(call, ast.Call):
            changed = False
            if isinstance(call.func, ast.Name) and call.func.id == "AsymmetricMASt3R":
                call.func = ast.Name(id="AsymmetricCroCo3DStereo", ctx=ast.Load())
                changed = True
            elif isinstance(call.func, ast.Attribute) and call.func.attr == "AsymmetricMASt3R":
                call.func.attr = "AsymmetricCroCo3DStereo"
                changed = True

            kept_keywords = []
            for kw in call.keywords:
                if kw.arg in _MAST3R_DROP_KWARGS:
                    changed = True
                    continue
                if (
                    kw.arg == "head_type"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value == "catmlp+dpt"
                ):
                    kw.value = ast.Constant(value="dpt")
                    changed = True
                if (
                    kw.arg == "output_mode"
                    and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)
                    and kw.value.value.startswith("pts3d+desc")
                ):
                    kw.value = ast.Constant(value="pts3d")
                    changed = True
                kept_keywords.append(kw)
            call.keywords = kept_keywords

            if changed:
                ast.fix_missing_locations(expr)
                return ast.unparse(expr), True
    except Exception:
        # Fall back to regex rewrite below.
        pass

    # Regex fallback for unexpected serialized arg forms.
    out = args.replace("AsymmetricMASt3R(", "AsymmetricCroCo3DStereo(")
    out = re.sub(r"head_type\s*=\s*['\"]catmlp\+dpt['\"]", "head_type='dpt'", out)
    out = re.sub(r"output_mode\s*=\s*['\"]pts3d\+desc\d+['\"]", "output_mode='pts3d'", out)
    for name in sorted(_MAST3R_DROP_KWARGS):
        out = re.sub(rf",\s*{re.escape(name)}\s*=\s*\([^)]*\)\s*", ", ", out)
        out = re.sub(rf",\s*{re.escape(name)}\s*=\s*[^,)]*\s*", ", ", out)
    out = re.sub(r",\s*,", ", ", out)
    out = re.sub(r",\s*\)", ")", out)
    return out, True


def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    args, was_mast3r = _rewrite_mast3r_args_for_dust3r(args)
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        if was_mast3r:
            print("detected MASt3R checkpoint args; using DUSt3R-compatible geometry-only adapter")
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


def _infer_arch_kwargs_from_safetensors(state_dict):
    def _max_block_index(prefix):
        pat = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
        idx = []
        for key in state_dict.keys():
            m = pat.match(key)
            if m:
                idx.append(int(m.group(1)))
        if not idx:
            raise RuntimeError(f"Cannot infer {prefix} depth from safetensors keys")
        return max(idx) + 1

    if "patch_embed.proj.weight" not in state_dict:
        raise RuntimeError("Invalid safetensors checkpoint: missing patch_embed.proj.weight")
    if "decoder_embed.weight" not in state_dict:
        raise RuntimeError("Invalid safetensors checkpoint: missing decoder_embed.weight")

    patch_weight = state_dict["patch_embed.proj.weight"]
    decoder_embed_weight = state_dict["decoder_embed.weight"]

    enc_embed_dim = int(patch_weight.shape[0])
    patch_size = int(patch_weight.shape[-1])
    dec_embed_dim = int(decoder_embed_weight.shape[0])
    enc_depth = _max_block_index("enc_blocks")
    dec_depth = _max_block_index("dec_blocks")
    enc_num_heads = max(int(enc_embed_dim // 64), 1)
    dec_num_heads = max(int(dec_embed_dim // 64), 1)

    # RoPE checkpoints do not carry enc/dec positional embedding buffers.
    if "enc_pos_embed" in state_dict:
        npos = int(state_dict["enc_pos_embed"].shape[0])
        grid = int(round(math.sqrt(float(npos))))
        img_size = grid * patch_size if grid * grid == npos else 512
        pos_embed = "cosine"
    else:
        img_size = 512
        pos_embed = "RoPE100"

    has_dpt_head = any(k.startswith("downstream_head1.dpt.") for k in state_dict.keys())
    head_type = "dpt" if has_dpt_head else "linear"

    # DPT head out_channels=4 => xyz + confidence.
    conf_mode = ("exp", 1, inf)
    out_head = state_dict.get("downstream_head1.dpt.head.4.weight")
    if out_head is not None and int(out_head.shape[0]) <= 3:
        conf_mode = None

    return dict(
        output_mode="pts3d",
        head_type=head_type,
        depth_mode=("exp", -inf, inf),
        conf_mode=conf_mode,
        landscape_only=False,
        patch_embed_cls="PatchEmbedDust3R",
        img_size=img_size,
        patch_size=patch_size,
        enc_embed_dim=enc_embed_dim,
        enc_depth=enc_depth,
        enc_num_heads=enc_num_heads,
        dec_embed_dim=dec_embed_dim,
        dec_depth=dec_depth,
        dec_num_heads=dec_num_heads,
        pos_embed=pos_embed,
    )


def load_model_safetensors(model_path, device, verbose=True):
    if verbose:
        print('... loading safetensors model from', model_path)
    try:
        from safetensors.torch import load_file
    except Exception as exc:
        raise ImportError(
            "Loading .safetensors requires safetensors package. "
            "Please install it in your environment."
        ) from exc

    state_dict = load_file(model_path, device='cpu')
    ctor_kwargs = _infer_arch_kwargs_from_safetensors(state_dict)
    if verbose:
        print("detected local safetensors checkpoint; instantiating from inferred geometry config")
        print(f"instantiating kwargs: {ctor_kwargs}")
    net = AsymmetricCroCo3DStereo(**ctor_kwargs)
    s = net.load_state_dict(state_dict, strict=False)
    if verbose:
        print(s)
    return net.to(device)


class AsymmetricCroCo3DStereo (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/junyi/monst3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            if str(pretrained_model_name_or_path).lower().endswith(".safetensors"):
                return load_model_safetensors(pretrained_model_name_or_path, device='cpu')
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_and_decoder': [self.mask_token, self.patch_embed, self.enc_blocks, self.dec_blocks, self.dec_blocks2],
        }
        freeze_all_params(to_be_frozen[freeze])
        print(f'Freezing {freeze} parameters')

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        if type(img_size) is int:
            img_size = (img_size, img_size)
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        # x (B, 576, 1024) pos (B, 576, 2); patch_size=16
        B,N,C = x.size()
        posvis = pos
        # add positional embedding without cls token
        assert self.enc_pos_embed is None
        # TODO: where to add mask for the patches
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, posvis)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]

        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        # warning! maybe the images have different portrait/landscape orientations
        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection
        original_D = f1.shape[-1]

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2
