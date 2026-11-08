from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_ORL(Method_CNN):
    def __init__(self, variant='baseline'):
        super().__init__([16,32] if variant=='baseline' else [16,32,64],
                         1,40,(7,5),hidden_dim=128,pool_kernel=2 if variant=='baseline' else 1)
