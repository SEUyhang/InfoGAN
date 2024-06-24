import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import random

from dataloader import get_data
from utils import *
from config import params

if(params['dataset'] == 'MNIST'):
    from models.mnist_model import Generator, Discriminator, DHead, QHead
elif(params['dataset'] == 'TEXBAT'):
    from models.texbat_model import Generator, Discriminator, DHead, QHead, CHead
elif(params['dataset'] == 'CelebA'):
    from models.celeba_model import Generator, Discriminator, DHead, QHead
elif(params['dataset'] == 'FashionMNIST'):
    from models.mnist_model import Generator, Discriminator, DHead, QHead

# Set random seed for reproducibility.
# seed = 1123
# seed = 3407
# seed = 42
seed = 2048
random.seed(seed)
torch.manual_seed(seed)       
print("Random Seed: ", seed)

# Use GPU if available.
device = torch.device("cuda:1" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

dataloader = get_data(params['dataset'], params['batch_size'], 'train')

# Set appropriate hyperparameters depending on the dataset used.
# The values given in the InfoGAN paper are used.
# num_z : dimension of incompressible noise.
# num_dis_c : number of discrete latent code used.
# dis_c_dim : dimension of discrete latent code.
# num_con_c : number of continuous latent code used.
if(params['dataset'] == 'MNIST'):
    params['num_z'] = 62
    params['num_dis_c'] = 1
    params['dis_c_dim'] = 10
    params['num_con_c'] = 2
elif(params['dataset'] == 'TEXBAT'):
    params['num_z'] = 256
    params['num_dis_c'] = 1
    params['dis_c_dim'] = 3
    params['num_con_c'] = 0
elif(params['dataset'] == 'CelebA'):
    params['num_z'] = 128
    params['num_dis_c'] = 10
    params['dis_c_dim'] = 10
    params['num_con_c'] = 0
elif(params['dataset'] == 'FashionMNIST'):
    params['num_z'] = 62
    params['num_dis_c'] = 1
    params['dis_c_dim'] = 10
    params['num_con_c'] = 2

# Plot the training images.
# sample_batch = next(iter(dataloader))
# plt.figure(figsize=(10, 10))
# plt.axis("off")
# plt.imshow(np.transpose(vutils.make_grid(
#     sample_batch[0].to(device)[ : 100], nrow=10, padding=2, normalize=True).cpu(), (1, 2, 0)))
# plt.savefig('Training Images {}'.format(params['dataset']))
# plt.close('all')

# Initialise the network.
log = torch.load("/home/yhang/GAN/InfoGAN-PyTorch/checkpoint/infoGan_TEXBAT/model_epoch_100_TEXBAT_learningrate_0")

netG = Generator().to(device)
discriminator = Discriminator().to(device)
netQ = QHead().to(device)
netC = CHead().to(device)
netG.load_state_dict(log["netG"])
netQ.load_state_dict(log["netQ"])
discriminator.load_state_dict(log["discriminator"])
# discriminator.apply(weights_init)
# netC.apply(weights_init)

# Loss for discrimination between real and fake images.
criterionD = nn.BCELoss()
# Loss for discrete latent code.
criterionQ_dis = nn.CrossEntropyLoss()
# Loss for continuous latent code.
criterionQ_con = NormalNLLLoss()

# Adam optimiser is used.
# optimD = optim.Adam([{'params': discriminator.parameters()}, {'params': netD.parameters()}], lr=params['D_learning_rate'] , betas=(params['beta1'], params['beta2']))
# optimG = optim.Adam([{'params': netG.parameters()}, {'params': netQ.parameters()}], lr=params['G_learning_rate'], betas=(params['beta1'], params['beta2']))
optimQ = optim.Adam([{'params': discriminator.parameters()}, {'params': netQ.parameters()}], lr=params['D_learning_rate'], betas=(params['beta1'], params['beta2']))
# Fixed Noise
# z = torch.randn(100, params['num_z'], 1, 1, device=device)
# fixed_noise = z
# if(params['num_dis_c'] != 0):
#     idx = np.arange(params['dis_c_dim']).repeat(10)
#     dis_c = torch.zeros(100, params['num_dis_c'], params['dis_c_dim'], device=device)
#     for i in range(params['num_dis_c']):
#         dis_c[torch.arange(0, 100), i, idx] = 1.0

#     dis_c = dis_c.view(100, -1, 1, 1)

#     fixed_noise = torch.cat((fixed_noise, dis_c), dim=1)

# if(params['num_con_c'] != 0):
#     con_c = torch.rand(100, params['num_con_c'], 1, 1, device=device) * 2 - 1
#     fixed_noise = torch.cat((fixed_noise, con_c), dim=1)

real_label = 1
fake_label = 0

# List variables to store results pf training.
C_losses = []

print("-"*25)
print("Starting Training Loop...\n")
print('Epochs: %d\nDataset: {}\nBatch Size: %d\nLength of Data Loader: %d'.format(params['dataset']) % (params['num_epochs'], params['batch_size'], len(dataloader)))
print("-"*25)

start_time = time.time()
iters = 0

for epoch in range(params['num_epochs'] * 2):
    epoch_start_time = time.time()

    for i, (data, label) in enumerate(dataloader, 0):
        # Get batch size
        b_size = data.size(0)
        # Transfer data tensor to GPU/CPU (device)
        real_data = data.to(device)
        label = label.to(device)

        
        # Updating Discriminator and QHead
        optimQ.zero_grad()
        
        # Real data Loss
        output1 = discriminator(real_data)
        probs_real, q_mu, q_var = netQ(output1)
        # Calculating loss for discrete latent code.
        target = torch.tensor(label.view(1, b_size), dtype=torch.long)
        # print('target:')
        # print(target)
        # print('probs_real:')
        # print(probs_real)
        dis_loss = 0
        for j in range(params['num_dis_c']):
            dis_loss += criterionQ_dis(probs_real[:, j*10 : j*10 + 10], target[j])

        # Calculating loss for continuous latent code.
        con_loss = 0
        if (params['num_con_c'] != 0):
            con_loss = criterionQ_con(noise[:, params['num_z']+ params['num_dis_c']*params['dis_c_dim'] : ].view(-1, params['num_con_c']), q_mu, q_var)*0.1
        # Calculate gradients.
        loss_real = dis_loss + con_loss
        loss_real.backward()

        # Fake data Loss
        noise, idx = noise_sample(params['num_dis_c'], params['dis_c_dim'], params['num_con_c'], params['num_z'], b_size, device)
        fake_data = netG(noise)
        output = discriminator(fake_data.detach())
        q_logits, q_mu, q_var = netQ(output)
        target = torch.LongTensor(idx).to(device)
        # Calculating loss for discrete latent code.
        dis_loss = 0
        for j in range(params['num_dis_c']):
            dis_loss += criterionQ_dis(q_logits[:, j*10 : j*10 + 10], target[j])

        # Calculating loss for continuous latent code.
        con_loss = 0
        if (params['num_con_c'] != 0):
            con_loss = criterionQ_con(noise[:, params['num_z']+ params['num_dis_c']*params['dis_c_dim'] : ].view(-1, params['num_con_c']), q_mu, q_var)*0.1
        
        loss_fake = dis_loss + con_loss
        # Calculate gradients.
        loss_fake.backward()
        C_loss = loss_real + loss_fake
        # Update parameters.
        optimQ.step()

        # Check progress of training.
        if i != 0 and i%100 == 0:
            print('[%d/%d][%d/%d]\tLoss_C: %.4f'
                  % (epoch+1, params['num_epochs'], i, len(dataloader), 
                    C_loss.item()))

        # Save the losses for plotting.
        C_losses.append(C_loss.item())

        iters += 1

    epoch_time = time.time() - epoch_start_time
    print("Time taken for Epoch %d: %.2fs" %(epoch + 1, epoch_time))
    # Generate image after each epoch to check performance of the generator. Used for creating animated gif later.
    # with torch.no_grad():
    #     gen_data = netG(fixed_noise).detach().cpu()
    # signal_list.append(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True))

    # Generate image to check performance of generator.
    # if((epoch+1) == 1 or (epoch+1) == params['num_epochs']/2):
    #     with torch.no_grad():
    #         gen_data = netG(fixed_noise).detach().cpu()
    #     plt.figure(figsize=(10, 10))
    #     plt.axis("off")
    #     plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1,2,0)))
    #     plt.savefig("Epoch_%d {}".format(params['dataset']) %(epoch+1))
    #     plt.close('all')

    # Save network weights.
    if (epoch+1) % params['save_epoch'] == 0:
        torch.save({
            'netG' : netG.state_dict(),
            'discriminator' : discriminator.state_dict(),
            'netQ' : netQ.state_dict(),
            'optimQ' : optimQ.state_dict(),
            'params' : params
            }, '/home/yhang/GAN/InfoGAN-PyTorch/checkpoint/classifier/model_epoch_%d_{}_learningrate_%d'.format(params['dataset']) %(epoch+1, params['D_learning_rate']))

training_time = time.time() - start_time
print("-"*50)
print('Training finished!\nTotal Time for Training: %.2fm' %(training_time / 60))
print("-"*50)

# Generate image to check performance of trained generator.
# with torch.no_grad():
#     gen_data = netG(fixed_noise).detach().cpu()
# plt.figure(figsize=(10, 10))
# plt.axis("off")
# plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1,2,0)))
# plt.savefig("Epoch_%d_{}".format(params['dataset']) %(params['num_epochs']))

# Plot the training losses.
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(C_losses,label="Classifier")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("Loss Curve {}".format(params['dataset']))

# Animation showing the improvements of the generator.
# fig = plt.figure(figsize=(10,10))
# plt.axis("off")
# ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
# anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
# anim.save('infoGAN_{}.gif'.format(params['dataset']), dpi=80, writer='imagemagick')
# plt.show()