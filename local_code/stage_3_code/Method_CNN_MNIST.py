from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_MNIST(Method_CNN):
    def __init__(self, variant='baseline'):
        super().__init__([16,32],1,10,(3,3),hidden_dim=128,
                         convolutions_per_block=2 if variant=='deeper' else 1,pool_kernel=2)
