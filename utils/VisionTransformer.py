import timm.models.vision_transformer

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
            
    def __call__(self, input):
        # Layer normalization
        input = self.norm(input)
        
        # Transformer blocks
        for blk in self.blocks:
            input = blk(input)
        
        # Normalize
        input = self.norm(input)
        
        return input