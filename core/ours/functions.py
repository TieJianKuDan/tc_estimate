import math

import torch
from torch.autograd import Variable


def gen_coord_3(batch_size, input_height, input_width):
    
    coords = torch.zeros(input_height, input_width, 2 * 3 * 3)
    
    parameters = torch.zeros(3)
    parameters[0] = batch_size
    parameters[1] = input_height
    parameters[2] = input_width
    
    center = torch.zeros(2)
    center[0]=torch.sub(torch.div(parameters[1], 2.0), 0.5)
    center[1]=torch.sub(torch.div(parameters[2], 2.0), 0.5)
    
    x_grid = torch.arange(0, parameters[1])
    y_grid = torch.arange(0, parameters[2]) 
    grid_x, grid_y = torch.meshgrid(x_grid, y_grid)
            
    delta_x = torch.sub(grid_x, center[0])
    delta_y = torch.sub(grid_y, center[1])
    PI = torch.mul(torch.Tensor([math.pi]), 2.0)
    theta=torch.atan2(delta_y, delta_x) % PI[0]
    theta=torch.round(10000.*theta)/10000.
    
    coords[:,:,0]=torch.add(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),0.0))),1.0)
    coords[:,:,1]=torch.add(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),0.0))),1.0)    
    
    coords[:,:,2]=torch.add(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),1.0))),1.0)
    coords[:,:,3]=torch.add(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),1.0))),0.0)
    
    coords[:,:,4]=torch.add(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),2.0))),1.0)
    coords[:,:,5]=torch.add(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),2.0))),-1.0)
    
    coords[:,:,6]=torch.add(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),3.0))),0.0)
    coords[:,:,7]=torch.add(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),3.0))),1.0)
    
    coords[:,:,10]=torch.add(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),4.0))),0.0)
    coords[:,:,11]=torch.add(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),4.0))),-1.0)
    
    coords[:,:,12]=torch.add(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),5.0))),-1.0)
    coords[:,:,13]=torch.add(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),5.0))),1.0)
    
    coords[:,:,14]=torch.add(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),6.0))),-1.0)
    coords[:,:,15]=torch.add(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),6.0))),0.0)
    
    coords[:,:,16]=torch.add(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),7.0))),-1.0)
    coords[:,:,17]=torch.add(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),7.0))),-1.0)
    
    coords=coords.expand(batch_size,-1,-1,-1) 
    coords=coords.permute(0, 3, 1, 2)
    coords = coords.cuda()
    
    return Variable(coords, requires_grad=False) 
