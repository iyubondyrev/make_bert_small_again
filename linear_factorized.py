import torch
from torch import nn

class FactorizedLinear(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.mat1 = nn.Parameter(torch.randn(output_dim, hidden_dim))
        self.mat2 = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.use_bias = bias

        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        output = x @ (self.mat1 @ self.mat2).T
        if self.use_bias and self.bias is not None:
            output += self.bias
        return output

    @classmethod
    def from_linear(cls, linear_layer, hidden_dim=64):
        output_dim, input_dim = linear_layer.weight.shape
        factorized_linear = cls(input_dim, output_dim, hidden_dim, bias=linear_layer.bias is not None)

        with torch.no_grad():
            u, s, v = torch.svd(linear_layer.weight)
            mat1 = u[:, :hidden_dim] @ torch.diag(s[:hidden_dim])
            mat2 = v[:, :hidden_dim].T
            
            factorized_linear.mat1.copy_(mat1)
            factorized_linear.mat2.copy_(mat2)

            if linear_layer.bias is not None:
                factorized_linear.bias.copy_(linear_layer.bias)

        return factorized_linear


def replace_linear_with_factorized(model, hidden_dim):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            output_dim, input_dim = module.weight.shape
            if input_dim > hidden_dim and output_dim > hidden_dim:
                setattr(model, name, FactorizedLinear.from_linear(module, hidden_dim))
                del module
            else:
                print(f"Skipping Linear layer {name} with dimensions {input_dim}x{output_dim}")
        elif isinstance(module, nn.Embedding):
            continue
        else:
            replace_linear_with_factorized(module, hidden_dim)