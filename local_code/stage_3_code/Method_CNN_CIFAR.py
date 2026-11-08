from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_CIFAR(Method_CNN):
    def __init__(self, variant='baseline'):
        super().__init__([32,64,128],3,10,(4,4),hidden_dim=256,
                         convolutions_per_block=2 if variant=='deeper' else 1)
