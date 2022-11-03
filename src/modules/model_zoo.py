from modules.layers_dsnet_anchor_based.dsnet_ab import DSNet
from modules.layers_dsnet_anchor_free.dsnet_af import DSNetAF
from modules.layers_casum.summarizer import CA_SUM
from modules.layers_pglsum.summarizer import PGL_SUM


def get_anchor_based(base_model, num_feature, num_hidden, anchor_scales,
                     num_head, **kwargs):
    return DSNet(base_model, num_feature, num_hidden, anchor_scales, num_head)


def get_anchor_free(base_model, num_feature, num_hidden, num_head, **kwargs):
    return DSNetAF(base_model, num_feature, num_hidden, num_head)


def get_ca_sum( **kwargs):
    return CA_SUM(input_size=1024, output_size=1024, block_size=60)


def get_pgl_sum( **kwargs):
    return PGL_SUM(input_size=1024, output_size=1024, num_segments=4, heads=8,
                                fusion="add", pos_enc="absolute")


def get_model(model_type, **kwargs):
    if model_type == 'dsnet_ab':
        return get_anchor_based(**kwargs)
    elif model_type == 'dsnet_af':
        return get_anchor_free(**kwargs)
    elif model_type == 'casum':
        return get_ca_sum(**kwargs)
    elif model_type == 'pglsum':
        return get_pgl_sum(**kwargs)
    else:
        raise ValueError('Invalid model type', model_type)
