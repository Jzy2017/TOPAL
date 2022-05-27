import os
import cv2
import numpy as np
import torch
import argparse
from model import WaterFusion
from DeepWB import deep_wb_single_task
import datetime

argparser = argparse.ArgumentParser()
argparser.add_argument('--epoch', type=int, help='epoch number', default=5)
argparser.add_argument('--lr', type=float, help='task-level inner update learning rate', default=1e-4)
argparser.add_argument('--bs', type=int, help='batch size', default=10)
argparser.add_argument('--logdir', type=str, default='logs/')
argparser.add_argument('--resume',action='store_true')
argparser.add_argument('--use_gpu', action='store_true')
argparser.add_argument('--ssim', type=float, default=300)
argparser.add_argument('--mse', type = float, default=20)
argparser.add_argument('--vgg', type = float, default=1)
argparser.add_argument('--egan', type=float, default=0.1)
argparser.add_argument('--w', type = float, default=None)
argparser.add_argument('--patchD_3', type = int, default=5)
args = argparser.parse_args()
WB_model = 'DeepWB/models'
MBD_model = 'networks/model'

test_data = 'Underwater/'

output_folder = 'Result/'

# ssh -L 18097:127.0.0.1:8097 jiangzhiying@172.31.73.75

def test(model1, model2, model3, input_path, save_path, w=None):
	save_path = save_path + args.logdir.split('/')[-1]+'_{}_{}_{}_{}'.format(args.ssim,args.mse, args.vgg, args.egan)
	print('saving at ',save_path)
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	names = os.listdir(input_path)

	device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
	starttime = datetime.datetime.now()
	for i, n in enumerate(names):
		print(i,'\t'+n)
		or_img = cv2.imread(input_path + n)
		img = cv2.cvtColor(or_img, cv2.COLOR_BGR2RGB)/255.0
		img = np.transpose(img,(2,0,1)).astype(np.float32)
		img = torch.from_numpy(img).unsqueeze(0).to(device)
		import time
		time_begin = time.time()

		pre1 = model1(img)
		pre2 = model2(img)
		output = model3(pre1, pre2, w)

		time_end = time.time()
		time = time_end - time_begin
		print('time:', time)

		output = torch.clamp(output, min=0, max=1)
		pre1 = (pre1 * 255).detach().cpu().numpy().astype(np.uint8)[0]
		
		output = (output * 255).detach().cpu().numpy().astype(np.uint8)[0]
		output = np.transpose(output, (1,2,0))
		output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
		
		cv2.imwrite(os.path.join(save_path, n), output)
	endtime = datetime.datetime.now()
	print(endtime - starttime)


def main():
	print('cuda ', torch.cuda.is_available())

	net_awb = deep_wb_single_task.deepWBnet()
	net_awb.load_state_dict(torch.load(os.path.join(WB_model, 'net_awb.pth')))
	net_awb.eval()
	MBD = torch.load(os.path.join(MBD_model, 'model.pkl'), map_location=lambda storage, loc: storage)
	MBD.eval()
	model = WaterFusion(in_channels= 3, out_channels = 3, num_features = 64, growthrate = 32)

	if args.use_gpu:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	else:
		device = torch.device('cpu')
	
	net_awb.to(device)
	MBD.to(device)

	optim = torch.optim.Adam(model.parameters(), lr = args.lr)
	
	filepath = args.logdir+'gan_{}_{}_{}_{}'.format(args.ssim,args.mse, args.vgg, args.egan)
	if not os.path.exists(filepath):
		os.makedirs(filepath)
	filepath = os.path.join(filepath, 'checkpoint_model.pt')
	
	resume_pt = 0

	if args.resume and os.path.exists(filepath):
		logs = torch.load(filepath)
		resume_pt = logs['epoch']
		model.load_state_dict(logs['state_dict'])


	model.load_state_dict(torch.load(filepath)['state_dict'])
	model.to(device)
	print('testing')
	if args.w:
		test(net_awb, MBD, model, test_data, output_folder, args.w)
	else:
		test(net_awb, MBD, model, test_data, output_folder)

if __name__ == '__main__':
	main()
