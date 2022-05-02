import torch
class AutoEncoder(torch.nn.Module):
    class Encoder(torch.nn.Module):
        class EncoderBlock(torch.nn.Module):
            def __init__(self, n_input, n_output):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(n_input, n_output, bias=False),
                    torch.nn.BatchNorm1d(n_output),
                    torch.nn.ReLU()
                )
            
            def forward(self, x):
                return self.net(x)            

        def __init__(self, dim_sizes = [32, 64], n_input = 20, n_output = 10):
            super().__init__()
            L = list()
            L.append(torch.nn.Linear(n_input, dim_sizes[0], bias = False))
            L.append(torch.nn.BatchNorm1d(dim_sizes[0]))
            L.append(torch.nn.ReLU())
            for i in range(1, len(dim_sizes)):
                L.append(self.EncoderBlock(dim_sizes[i-1], dim_sizes[i]))
            L.append(torch.nn.Linear(dim_sizes[-1], n_output))
            self.layers = torch.nn.Sequential(*L)
        
        def forward(self, x):
            return self.layers(x)

    class Decoder(torch.nn.Module):
        class DecoderBlock(torch.nn.Module):
            def __init__(self, n_input, n_output):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(n_input, n_output, bias=False),
                    torch.nn.BatchNorm1d(n_output),
                    torch.nn.ReLU()
                )
            
            def forward(self, x):
                return self.net(x)            

        def __init__(self, dim_sizes = [32, 64], n_input = 10, n_output = 20):
            super().__init__()
            L = list()
            L.append(torch.nn.Linear(n_input, dim_sizes[0], bias = False))
            L.append(torch.nn.BatchNorm1d(dim_sizes[0]))
            L.append(torch.nn.ReLU())
            for i in range(1, len(dim_sizes)):
                L.append(self.DecoderBlock(dim_sizes[i-1], dim_sizes[i]))
            L.append(torch.nn.Linear(dim_sizes[-1], n_output))
            self.layers = torch.nn.Sequential(*L)
        
        def forward(self, x):
            return self.layers(x)

    def __init__(self, encoder_dim_sizes = [32, 64], decoder_dim_sizes = [32, 64], n_input = 20, latent_dim = 10):
            super().__init__()
            self.encoder = self.Encoder(dim_sizes=encoder_dim_sizes, n_input=n_input, n_output=latent_dim)
            self.decoder = self.Decoder(dim_sizes=decoder_dim_sizes, n_input=latent_dim, n_output=n_input)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AutoEncoderConv(torch.nn.Module):
    class Encoder(torch.nn.Module):
        class EncoderBlock(torch.nn.Module):
            def __init__(self, n_input, n_output, kernel_size = 3, padding = 1, stride = 1):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Conv1d(n_input, n_output, kernel_size = kernel_size, padding = padding, stride = stride, bias=False),
                    torch.nn.BatchNorm1d(n_output),
                    torch.nn.ReLU(),
                    torch.nn.Conv1d(n_output, n_output, kernel_size = kernel_size, padding = padding, stride = 1, bias=False),
                    torch.nn.BatchNorm1d(n_output),
                    torch.nn.ReLU()
                )
                self.downsample = None
                if n_input != n_output or stride != 1:
                    self.downsample = torch.nn.Sequential(
                        torch.nn.Conv1d(n_input, n_output, kernel_size = 1, stride = stride, bias = False),
                        torch.nn.BatchNorm1d(n_output)
                    )
            
            def forward(self, x):
                z = x
                if self.downsample != None:
                    z = self.downsample(x)
                return self.net(x) + z

        def __init__(self, latent_dim = 1):
            super().__init__()
            L = list()
            L.append(self.EncoderBlock(1, 32, kernel_size = 7, padding = 3, stride=2))
            L.append(self.EncoderBlock(32, 64, kernel_size = 3, padding = 1, stride=1))
            L.append(self.EncoderBlock(64, 128, kernel_size = 3, padding = 1, stride=1))
            L.append(self.EncoderBlock(128, 128, stride=1))
            L.append(self.EncoderBlock(128, 64, kernel_size = 3, padding = 1, stride=1))
            L.append(self.EncoderBlock(64, 32, kernel_size = 3, padding = 1, stride=1))
            # L.append(torch.nn.Conv1d(32, 1, kernel_size = 3, padding = 1, stride=1))
            L.append(self.EncoderBlock(32, 1, kernel_size = 3, padding = 1, stride=1))
            L.append(torch.nn.Linear(10, latent_dim))

            self.layers = torch.nn.Sequential(*L)
        
        def forward(self, x):
            return self.layers(x)

    class Decoder(torch.nn.Module):
        class DecoderBlock(torch.nn.Module):
            def __init__(self, n_input, n_output, kernel_size = 3, padding = 1, stride=1):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Conv1d(n_input, n_output, kernel_size = kernel_size, padding = padding, stride = stride, bias=False),
                    torch.nn.BatchNorm1d(n_output),
                    torch.nn.ReLU(),
                    torch.nn.Conv1d(n_output, n_output, kernel_size = kernel_size, padding = padding, stride = 1, bias=False),
                    torch.nn.BatchNorm1d(n_output),
                    torch.nn.ReLU()
                )
                self.downsample = None
                if n_input != n_output or stride != 1:
                    self.downsample = torch.nn.Sequential(
                        torch.nn.Conv1d(n_input, n_output, kernel_size = 1, stride = stride, bias = False),
                        torch.nn.BatchNorm1d(n_output)
                    )
            
            def forward(self, x):
                z = x
                if self.downsample != None:
                    z = self.downsample(x)
                return self.net(x) + z

        class TransposeBlock(torch.nn.Module):
            def __init__(self, n_input, n_output, kernel_size = 3, padding = 1, stride = 2, output_padding=1):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.ConvTranspose1d(n_input, n_output, kernel_size = kernel_size, padding = padding, stride = stride, output_padding=output_padding, bias=False),
                    torch.nn.BatchNorm1d(n_output),
                    torch.nn.ReLU()
                )
            
            def forward(self, x):
                return self.net(x)

        def __init__(self, latent_dim = 1):
            super().__init__()
            L = list()
            L.append(torch.nn.Linear(latent_dim, 10, bias=False))
            L.append(torch.nn.BatchNorm1d(1))
            L.append(torch.nn.ReLU())
            L.append(self.TransposeBlock(1, 32, kernel_size = 3, padding = 1, stride = 2, output_padding=1))
            L.append(self.DecoderBlock(32, 64, kernel_size = 3, padding = 1, stride=1))
            L.append(self.DecoderBlock(64, 128, kernel_size = 3, padding = 1, stride=1))
            L.append(self.DecoderBlock(128, 128, kernel_size = 3, padding = 1))
            L.append(self.DecoderBlock(128, 64, kernel_size = 3, padding = 1, stride=1))
            L.append(self.DecoderBlock(64, 32, kernel_size = 3, padding = 1, stride=1))
            L.append(self.DecoderBlock(32, 32, kernel_size = 3, padding = 1, stride=1))
            # L.append(torch.nn.Conv1d(32, 1, kernel_size = 3, padding = 1, stride=1))
            L.append(self.DecoderBlock(32, 1, kernel_size = 3, padding = 1, stride=1))
            L.append(torch.nn.Linear(20, 20))

            self.layers = torch.nn.Sequential(*L)
        
        def forward(self, x):
            return self.layers(x)

    def __init__(self, latent_dim = 1):
            super().__init__()
            self.encoder = self.Encoder(latent_dim)
            self.decoder = self.Decoder(latent_dim)
        
    def forward(self, x):
        encoded = self.encoder(x[:, None, :])
        decoded = self.decoder(encoded)
        return decoded.squeeze()

class Predictor(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(n_input, n_output, bias=False),
                torch.nn.BatchNorm1d(n_output),
                torch.nn.ReLU()
            )
        
        def forward(self, x):
            return self.net(x)            

    def __init__(self, dim_sizes = [64, 128, 256, 256, 128, 64], n_input = 2, n_output = 1):
        super().__init__()
        L = list()
        L.append(torch.nn.Linear(n_input, dim_sizes[0], bias = False))
        L.append(torch.nn.BatchNorm1d(dim_sizes[0]))
        L.append(torch.nn.ReLU())
        for i in range(1, len(dim_sizes)):
            L.append(self.Block(dim_sizes[i-1], dim_sizes[i]))
        L.append(torch.nn.Linear(dim_sizes[-1], n_output))
        self.layers = torch.nn.Sequential(*L)
    
    def forward(self, x):
        return self.layers(x)

def save_model(model, p):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), p))


def load_model(p, m):
    from torch import load
    from os import path
    r = m()
    r.load_state_dict(load(p, map_location='cpu'))
    return r