device = torch.device("cuda:0" if (torch.cuda.is_available() and args["gpu"] > 0) else "cpu")
	netG = Generator()
	print(netG)
	netD = Discriminator()
	print(netD)
	checkpoint = torch.load(os.path.join(args["checkpoint_dir"],"generator_wgan_pc.pkl"),
							map_location = lambda storage, loc:storage)
	print(checkpoint.keys())
	netG.load_state_dict(checkpoint)
	netG.to(device)
	del checkpoint
	checkpoint = torch.load(os.path.join(args["checkpoint_dir"],"discriminator_wgan_pc.pkl"),
							map_location = lambda storage, loc:storage)
	netD.load_state_dict(checkpoint)
	del checkpoint
	netD.to(device)
	netG.eval()
	netD.eval()
