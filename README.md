# CA-Dense-UNet
An unofficial code reproduction of **Channel Attention Dense U-Net for Multichannel Speech Enhancement**[1]

[1]Tolooshams B, Giri R, Song A H, et al. Channel-attention dense u-net for multichannel speech enhancement[C]//ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020: 836-840.

<div align="center"><img src="./doc/donw up block.png" width="400"></div>

It was observed that the number of output channels for each Down block is 4 times that of the input, while the number of output channels for Up block is 1/2 of the input, which leads to a mismatch in the number of network channels during operation

<div align="center"><img src="./doc/u-net.png" width="400" ></div>

According to the above logic, the output of the last Down block in the above figure should be 16K, but the author wrote it as 8K

If you want the network to operate normally, you need to add a 1 after concatenating the CA output of Up block × The convolutional block of 1 is used to adjust the number of channels, so that the output channel number of Up block is 1/4 of the input, which can ensure the normal operation of the network with minimal modifications. This is how the changes are made in the code

Unfortunately, my code proficiency is limited and I am unable to perfectly reproduce the results in the author's paper. I am now publishing the code I have edited in the hope of helping others who wish to reproduce it
