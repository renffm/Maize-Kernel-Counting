"""BiFPN+PRM-only model entry point."""

from .p2pnet_bifpn import build as build_bifpn_prm


def build_model(args, training=False):
    """Build the BiFPN+PRM variant regardless of generic CLI defaults."""
    args.use_bifpn = True
    args.use_prm = True
    return build_bifpn_prm(args, training)
