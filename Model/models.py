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
            L.append(torch.nn.BatchNorm1d(n_input))
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
            L.append(torch.nn.BatchNorm1d(n_input))
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
            def __init__(self, n_input, n_output, stride = 1):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Conv1d(n_input, n_output, kernel_size = 3, padding = 1, stride = stride, bias=False),
                    torch.nn.BatchNorm1d(n_output),
                    torch.nn.ReLU()
                )
            
            def forward(self, x):
                return self.net(x)            

        def __init__(self, dim_sizes = [32, 64, 32], n_input = 1, latent_dim = 10):
            super().__init__()
            L = list()
            L.append(torch.nn.BatchNorm1d(n_input))
            L.append(torch.nn.Conv1d(n_input, dim_sizes[0], kernel_size = 7, padding = 3, stride = 1, bias=False))
            L.append(torch.nn.BatchNorm1d(dim_sizes[0]))
            L.append(torch.nn.ReLU())
            for i in range(1, len(dim_sizes)):
                L.append(self.EncoderBlock(dim_sizes[i-1], dim_sizes[i], stride=1))
            L.append(self.EncoderBlock(dim_sizes[-1], 1, stride=2))
            self.layers = torch.nn.Sequential(*L)
        
        def forward(self, x):
            return self.layers(x)

    class Decoder(torch.nn.Module):
        class DecoderBlock(torch.nn.Module):
            def __init__(self, n_input, n_output, stride = 1):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Conv1d(n_input, n_output, kernel_size = 3, padding = 1, stride = stride, bias=False),
                    torch.nn.BatchNorm1d(n_output),
                    torch.nn.ReLU()
                )
            
            def forward(self, x):
                return self.net(x)            

        def __init__(self, dim_sizes = [32, 64, 32], latent_dim = 10, n_output = 20):
            super().__init__()
            L = list()
            L.append(torch.nn.BatchNorm1d(1))
            L.append(torch.nn.ConvTranspose1d(1, dim_sizes[0], kernel_size = 7, padding = 3, stride = 1, bias=False))
            L.append(torch.nn.BatchNorm1d(dim_sizes[0]))
            L.append(torch.nn.ReLU())
            for i in range(1, len(dim_sizes)):
                L.append(self.DecoderBlock(dim_sizes[i-1], dim_sizes[i]))
            L.append(torch.nn.ConvTranspose1d(dim_sizes[-1], 1, kernel_size = 3, padding = 1, stride = 2, output_padding=1))
            self.layers = torch.nn.Sequential(*L)
        
        def forward(self, x):
            return self.layers(x)

    def __init__(self, encoder_dim_sizes = [32, 64, 32], decoder_dim_sizes = [32, 64, 32], n_input = 1, latent_dim = 1):
            super().__init__()
            self.encoder = self.Encoder(dim_sizes=encoder_dim_sizes, n_input=n_input, latent_dim=latent_dim)
            self.decoder = self.Decoder(dim_sizes=decoder_dim_sizes, latent_dim=latent_dim, n_output=n_input)
        
    def forward(self, x):
        encoded = self.encoder(x[:, None, :])
        decoded = self.decoder(encoded)
        return decoded.squeeze()

def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, AutoEncoder):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'autoencoder.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = AutoEncoder()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'autoencoder.th'), map_location='cpu'))
    return r