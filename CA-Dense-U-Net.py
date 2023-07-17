import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt



class Encoder(nn.Module):
    def __init__(self, n_fft, hop_length=None, win_length=None, window=None, center=True):
        super(Encoder, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center

        if window is not None:
            if isinstance(window, str):
                # 根据字符串选择窗口类型
                if window.lower() == "hanning":
                    self.window_fn = torch.hann_window(self.win_length)
                elif window.lower() == "hamming":
                    self.window_fn = torch.hamming_window(self.win_length)
                elif window.lower() == "blackman":
                    self.window_fn = torch.blackman_window(self.win_length)
                else:
                    raise ValueError("Invalid window type. Supported types are 'hanning', 'hamming', and 'blackman'.")
            elif isinstance(window, torch.Tensor):
                self.window_fn = window
            else:
                raise ValueError("Invalid window type. It should be either a string or a torch.Tensor.")
        else:
            self.window_fn = None
        

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.window_fn = self.window_fn.to(device=self.device)
    
    
    
    def forward(self, input_signal):
        # 输入信号的形状：(batch_size, channels, time)
        
        batch_size, channels, time = input_signal.size()

        #self.window_fn = self.window_fn.clone().detach().to(device=device)
        self.device = input_signal.device
        
        self.window_fn = self.window_fn.clone().detach().to(device=self.device)
        # STFT变换
        complex_spectrum = torch.stft(input_signal.view(batch_size*channels,time), n_fft=self.n_fft, hop_length=self.hop_length,
                                      win_length=self.win_length, window=self.window_fn, center=self.center,
                                      return_complex=True)
        # 拼接实部和虚部
        _, Fre, Len = complex_spectrum.size()        # B*M , F= num freqs, L= num frame, 2= real imag

        complex_spectrum = complex_spectrum.view([batch_size, channels, Fre, Len])      # B*M, F, L -> B, M, F, L
        
        complex_spectrum = torch.cat([complex_spectrum.real,complex_spectrum.imag],dim=1) # B, 2*M, L, F
        
        #丢弃fre最后一个bin
        complex_spectrum = complex_spectrum[:, :, :Fre-1, :]  
    

        return complex_spectrum


class Decoder(nn.Module):
    def __init__(self, n_fft, hop_length=None, win_length=None, window=None, center=True):
        super(Decoder, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center

        if window is not None:
            if isinstance(window, str):
                # 根据字符串选择窗口类型
                if window.lower() == "hanning":
                    self.window_fn = torch.hann_window(self.win_length)
                elif window.lower() == "hamming":
                    self.window_fn = torch.hamming_window(self.win_length)
                elif window.lower() == "blackman":
                    self.window_fn = torch.blackman_window(self.win_length)
                else:
                    raise ValueError("Invalid window type. Supported types are 'hanning', 'hamming', and 'blackman'.")
            elif isinstance(window, torch.Tensor):
                self.window_fn = window
            else:
                raise ValueError("Invalid window type. It should be either a string or a torch.Tensor.")
        else:
            self.window_fn = None
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.window_fn = self.window_fn.to(device=self.device)


    def forward(self, complex_spectrum_concat, time):
        # 输入复数谱的形状：(batch_size, 2*channels, F, T)
        batch_size, d_channels, Fre, Len = complex_spectrum_concat.size()
        
        
        #将丢失的频率bin补零
        
        complex_spectrum_concat = torch.cat([complex_spectrum_concat, torch.zeros(batch_size, d_channels, 1, Len).to(device=self.device)], dim=2)
        #重新计算size()
        batch_size, d_channels, Fre, Len = complex_spectrum_concat.size()

        # 拆分实部和虚部
        real_part = complex_spectrum_concat[:,:d_channels//2,:,:]
        imag_part = complex_spectrum_concat[:,d_channels//2:,:,:]
        complex_spectrum = torch.complex(real_part, imag_part)

        complex_spectrum = complex_spectrum.view([-1,Fre,Len]) 
        
        self.device = complex_spectrum.device
        self.window_fn = self.window_fn.clone().detach().to(device=self.device)
        # 进行逆STFT变换
        reconstructed_signal = torch.istft(complex_spectrum, n_fft=self.n_fft, hop_length=self.hop_length,
                                           win_length=self.win_length, window=self.window_fn, length=time, center=self.center
                                           )
        
        reconstructed_signal = reconstructed_signal.view([batch_size, d_channels//2 ,time]) # channels = d_channels // 2

        
        return reconstructed_signal
    

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate=1, num_layers = 4):
        super(DenseBlock, self).__init__()


        layers = []
        for i in range(num_layers):
            # 计算当前卷积层的输入通道数
            input_channels = in_channels + i * growth_rate

            # 创建当前卷积层
            layer = nn.Sequential(
                
                
                nn.Conv2d(input_channels, growth_rate, kernel_size=3, padding=1),
                nn.BatchNorm2d(growth_rate),
                nn.ELU(inplace=True),      
                      
            )

            # 将当前卷积层添加到列表中
            layers.append(layer)

        # 将所有卷积层组合成一个模块
        self.layers = nn.ModuleList(layers)

        # 计算最终输出通道数
        self.DSout_channels = in_channels + num_layers * growth_rate

        # 1x1卷积层来调整通道数
        self.conv1x1 = nn.Conv2d(self.DSout_channels, out_channels, kernel_size=1)

    
    def forward(self, x):
        features = [x]

        for layer in self.layers:
            # 每个卷积层的输入是所有前面层的输出的拼接
            input_tensor = torch.cat(features, dim=1)

            # 计算当前卷积层的输出
            output = layer(input_tensor)

            # 将当前卷积层的输出添加到特征列表中
            features.append(output)

        # 将所有层的输出拼接起来作为最终输出
        output = torch.cat(features, dim=1)

        # 使用1x1卷积层调整通道数
        output = self.conv1x1(output)

        return output
    
class ChannelsAttentionModule(nn.Module):
    def __init__(self, T, d):
        super(ChannelsAttentionModule, self).__init__()
        self.k_conv = nn.Conv2d(T, d, kernel_size=1)
        self.q_conv = nn.Conv2d(T, d, kernel_size=1)
        self.v_conv = nn.Conv2d(T, T, kernel_size=1)

    
    def forward(self, x):
        batch_size, Channels, Fre, T = x.size()

        x_real_part = x[:,:Channels//2,:,:]
        x_imag_part = x[:,Channels//2:,:,:]
        # x = torch.complex(x_real_part, x_imag_part)
        
        # k(x) 
        # (batch_size, Channels, Fre, T) -->  (batch_size, T, Channels, Fre)
        # (0,1,2,3) -> (0,3,1,2)

        k_r = self.k_conv(x_real_part.permute(0,3,1,2))
        k_i = self.k_conv(x_imag_part.permute(0,3,1,2)) 
        # k = torch.exp(k)  # Exponential non-linearity 

        # q(x) 
        # (batch_size, Channels, Fre, T) -->  (batch_size, T, Channels, Fre)
        # (0,1,2,3) -> (0,3,1,2)
        #q = self.q_conv(x.permute(0,3,1,2))
        q_r = self.q_conv(x_real_part.permute(0,3,1,2))
        q_i = self.q_conv(x_imag_part.permute(0,3,1,2))  
        # q = torch.exp(q)  # Exponential non-linearity 

        # v(x)
        # (batch_size, T, Channels, Fre)   x.permute(0,3,1,2)
        #v = self.v_conv(x.permute(0,3,1,2))
        v_r = self.v_conv(x_real_part.permute(0,3,1,2))
        v_i = self.v_conv(x_imag_part.permute(0,3,1,2)) 

        # Compute similarity matrix P
        # (batch_size, d, Channels, Fre) -> (batch_size, Fre, d, Channels)
        # (0,1,2,3) -> (0,3,1,2)
        k_r = k_r.permute(0,3,1,2)
        k_i = k_i.permute(0,3,1,2)

        q_r = q_r.permute(0,3,1,2)
        q_i = q_i.permute(0,3,1,2)

        k = torch.complex(k_r, k_i)
        q = torch.complex(q_r, q_i)
        
        
        # (batch_size, Fre, Channels, Channels)
        P = torch.matmul(k.transpose(2, 3), q) 

        # Compute attention weights W
        #W = torch.softmax(P, dim=2) # (batch_size, F, Channels_N, Channels)
        real_part = P.real
        imag_part = P.imag
        
        # 对实部和虚部分别进行softmax
        real_softmax = torch.softmax(real_part, dim=2)
        imag_softmax = torch.softmax(imag_part, dim=2)
        
        # 合并实部和虚部
        W = torch.complex(real_softmax, imag_softmax)

        # Multiply attention weights with v(x)
        # (batch_size, T, Channels, Fre) -> (batch_size, Fre, T, Channels)
        # (0,1,2,3) -> (0,3,1,2)
        v_r = v_r.permute(0,3,1,2)
        v_i = v_i.permute(0,3,1,2)
        v = torch.complex(v_r, v_i)

        #output = torch.matmul(v,W) # (batch_size, Fre, T, Channels)*(batch_size, Fre, Channels_N, Channels)
        output = torch.matmul(v,W) # (batch_size, Fre, T, Channels)*(batch_size, Fre, Channels_N, Channels)

        # (batch_size, Fre, T, Channels) -> (batch_size, Channels, Fre, T)
        # (0,1,2,3) -> (0,3,1,2)
        output = output.permute(0,3,1,2)

        output = torch.cat([output.real,output.imag],dim=1)

        return output






class DownsamplingModule(nn.Module):
    def __init__(self, DS_in_channels, DS_growth_rate, DS_num_layers, CA_frame_length, CA_d):
        super(DownsamplingModule, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dense_block = DenseBlock(
                DS_in_channels, 
                DS_in_channels *2 ,
                growth_rate = DS_growth_rate, 
                num_layers = DS_num_layers
                )
        
        self.channels_attention = ChannelsAttentionModule(CA_frame_length, CA_d)


    def forward(self, x):
        pooled = self.pool(x)
        dense_block_output = self.dense_block(pooled)
        channels_attention_output = self.channels_attention(dense_block_output)
        output = torch.cat([dense_block_output, channels_attention_output], dim=1)

        return output


class UpsamplingModule(nn.Module):
    def __init__(self, DS_out_channels, DS_growth_rate, DS_num_layers, CA_frame_length, CA_d):
        super(UpsamplingModule, self).__init__()

        self.transposed_conv = nn.ConvTranspose2d(
            DS_out_channels * 4, DS_out_channels * 4, 
            kernel_size=3, stride=2, padding=1, output_padding=1
            )
        
        self.conv = nn.Conv2d(DS_out_channels * 4, DS_out_channels *1, kernel_size=3, padding=1)
        self.dense_block = DenseBlock(
            2* DS_out_channels,
            1* DS_out_channels,  # 改为从4k -> 1k(原本为2k)
            growth_rate = DS_growth_rate, 
            num_layers = DS_num_layers
            )

        self.channels_attention = ChannelsAttentionModule(CA_frame_length,CA_d)

        #加入1*1的卷积
        self.conv_c = nn.Sequential(
                nn.Conv2d(2* DS_out_channels, 1* DS_out_channels, kernel_size=1),
                nn.BatchNorm2d(DS_out_channels),
                nn.ReLU(inplace=True),
                
            )

    def forward(self, x, skip_connection):
        output = self.transposed_conv(x)
        output = self.conv(output)

        # 上采样后的输出与skip connection拼接
        output = torch.cat([output, skip_connection], dim=1)

        output = self.dense_block(output)
        #output = self.conv_afterDS(output)
        output = torch.cat([output, self.channels_attention(output)], dim=1)
        output = self.conv_c(output)

        return output
    



class MaskEstimationNetwork(nn.Module):
    def __init__(self, n_fft, hop_length, win_length, window, center, K, d ,sample_length, channels, U_net_layers):
        super(MaskEstimationNetwork, self).__init__()
        self.encoder = Encoder(n_fft, hop_length, win_length, window)
        #计算帧长
        self.sample_length = sample_length

        frame_length = (sample_length) // hop_length  + 1

        self.channels_attention = ChannelsAttentionModule(T = frame_length, d = d)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        #nn.MaxPool2d(kernel_size=2, stride=2)
        self.downsampling_modules = nn.ModuleList([
            DownsamplingModule(
                DS_in_channels = int((4**i)*K), 
                DS_growth_rate = 1, 
                DS_num_layers = 6,
                CA_frame_length= int(frame_length // (2**(i+1))), 
                CA_d = d 
                ) 
            for i in range(U_net_layers)
        ])

        
        self.dense_block = DenseBlock(channels*4, K, growth_rate=4, num_layers=24)

        self.upsampling_modules = nn.ModuleList([
            UpsamplingModule(
                DS_out_channels = int((4**i)*K), 
                DS_num_layers = 6, 
                DS_growth_rate = 1, 
                CA_frame_length = int(frame_length // (2**i)),  
                CA_d = d
                ) 
            for i in range(U_net_layers)
        ])

        self.mask_estimation_conv = nn.Conv2d(K, 2*channels, kernel_size=1)

        self.decoder = Decoder(n_fft, hop_length, win_length, window, center)

        

    def forward(self, input_signal):
        # Encoder
        complex_spectrum = self.encoder(input_signal)

        # cat (complex_spectrum, Channels Attention Module)
        complex_spectrum_CA = self.channels_attention(complex_spectrum)
        complex_spectrum_CA = torch.cat([complex_spectrum,complex_spectrum_CA], dim=1)

        # Dense Block
        dense_block_output = self.dense_block(complex_spectrum_CA)

        # Downsample
        skip_connections = []
        for i, downsampling_module in enumerate(self.downsampling_modules):
            skip_connections.append(dense_block_output)
            dense_block_output = downsampling_module(dense_block_output)

        

        # Upsample
        for i, upsampling_module in reversed(list(enumerate(self.upsampling_modules))):
            dense_block_output = upsampling_module(dense_block_output, skip_connections[i])

        # Mask Estimation
        singal_mask = self.mask_estimation_conv(dense_block_output)

        # Spectrogram Masking
        singal_spectrum = complex_spectrum * singal_mask

        _, channels, _, _ = singal_mask.size()

        # 拆分实部和虚部
        S_real_part = singal_mask[:,:channels//2,:,:]
        S_imag_part = singal_mask[:,channels//2:,:,:]

        N_real_part = 1 - S_real_part
        N_imag_part = -S_imag_part
        noise_mask = torch.cat([N_real_part,N_imag_part],dim=1) # B, 2*M, L, F


        noise_spectrum = complex_spectrum * noise_mask


        # Decoder
        reconstructed_signal = self.decoder(singal_spectrum, self.sample_length)
        reconstructed_noise  = self.decoder(noise_spectrum, self.sample_length)
        
        return reconstructed_signal,reconstructed_noise



#测试部分
if __name__ == "__main__":
    # 创建一个测试输入信号
    batch_size = 2
    channels = 4
    time = 16128 #使得计算后的帧长符合2的倍数，帧长计算：frame_length = (sample_length) // hop_length  + 1
    input_signal = torch.randn(batch_size, channels, time)

    # 实例化MaskEstimationNetwork
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    window = "hanning"
    center = True
    K = 16
    d = 20
    sample_length = time
    U_net_layers = 3

    model = MaskEstimationNetwork(n_fft, hop_length, win_length, window, center, K, d, sample_length, channels, U_net_layers)

    # 运行前向传播
    reconstructed_signal, reconstructed_noise = model(input_signal)

    # 输出结果形状
    print("Reconstructed Signal Shape:", reconstructed_signal.shape)
    print("Reconstructed Noise Shape:", reconstructed_noise.shape)


