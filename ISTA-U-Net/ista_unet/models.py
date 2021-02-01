import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def relu(x, lambd):
    return nn.functional.relu(x - lambd)

def initialize_sparse_codes(y, hidden_layer_width_list, rand_bool = False):
    num_layers = len(hidden_layer_width_list)
    code_list = []

    num_samples =  y.shape[0]    
    input_spatial_dim_1 = y.shape[2]
    input_spatial_dim_2 = y.shape[3]
    
    if rand_bool:
        initializer = torch.rand
    else:
        initializer = torch.zeros
        
    for i in range(num_layers):
        feature_map_dim_1 = int(input_spatial_dim_1/  (2 ** i) )
        feature_map_dim_2 = int(input_spatial_dim_2/  (2 ** i) )
        code_tensor = initializer(num_samples, hidden_layer_width_list[num_layers-i-1],  feature_map_dim_1, feature_map_dim_2 )
        code_list.append(code_tensor)
        
    code_list.reverse() # order the code from low-spatial-dim to high-spatial-dim.
    return code_list

def power_iteration_conv_model(conv_model, num_simulations: int):
        
    hidden_layer_width_list = conv_model.hidden_layer_width_list
    
    eigen_vec_list = initialize_sparse_codes(y = torch.zeros(1, 3, 64, 64), hidden_layer_width_list = hidden_layer_width_list, rand_bool = True)
    
    adjoint_conv_model = adjoint_dictionary_model(conv_model)
    
    eigen_vec_list = [ x for x in eigen_vec_list ]
    
    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        eigen_vec_list = adjoint_conv_model(conv_model(eigen_vec_list))
        # calculate the norm
        flatten_x_norm = torch.norm(torch.cat([x.flatten() for x in eigen_vec_list ]) )
        # re-normalize the vector
        eigen_vec_list = [x/ flatten_x_norm for x in eigen_vec_list] 
        
    eigen_vecs_flatten = torch.cat([x.flatten() for x in eigen_vec_list])
    
    linear_trans_eigen_vecs_list = adjoint_conv_model(conv_model(eigen_vec_list ))
    
    linear_trans_eigen_vecs_list_flatten = torch.cat([x.flatten() for x in linear_trans_eigen_vecs_list] )
    
    numerator = torch.dot(eigen_vecs_flatten, linear_trans_eigen_vecs_list_flatten)
    
    denominator = torch.dot(eigen_vecs_flatten, eigen_vecs_flatten)
    
    eigenvalue = numerator / denominator
    return eigenvalue

    

class adjoint_conv_op(nn.Module):
    # The adjoint of a conv module.
    def __init__(self, conv_op):
        super().__init__()
        in_channels = conv_op.out_channels
        out_channels = conv_op.in_channels
        kernel_size = conv_op.kernel_size
        padding = kernel_size[0] // 2

        # transpose convolution 
        self.transpose_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding= padding, bias= False)
        
        # tie the weights of transpose convolution with convolution 
        self.transpose_conv.weight = conv_op.weight

    def forward(self, x):
        return self.transpose_conv(x)
    

class up_block(nn.Module):
    """
    A module that contains:
    (1) an up-sampling operation (implemented by bilinear interpolation or upsampling)
    (2) convolution operations
    """
        
    def __init__(self, kernel_size, in_channels, out_channels, bilinear_bool = False, bias_bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias_bool = bias_bool
        self.bilinear_bool = bilinear_bool
        
        # the up-sampling operation
        if bilinear_bool:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(in_channels, in_channels // 2, kernel_size = 3, bias= self.bias_bool))
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2, bias= self.bias_bool)
        
        # the 2d convolution operation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding= kernel_size // 2, bias= self.bias_bool)
 
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # input is CHW
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    

class adjoint_up_block(nn.Module):
    # adjoint of up_block module
    
    def __init__(self, up_block_model):
        super().__init__()
        
        # to construct the adjoint model, one should exclude additive biases and use transposed conv for upsampling.
        assert up_block_model.bias_bool == False
        assert up_block_model.bilinear_bool == False
        
        in_channels = up_block_model.out_channels
        out_channels = up_block_model.in_channels
        
        self.adjoint_conv_op = adjoint_conv_op(up_block_model.conv)
        self.adjoint_up =  nn.Conv2d(in_channels , in_channels // 2, kernel_size=2, stride=2, bias= False)
        self.adjoint_up.weight = up_block_model.up.weight
        
        
    def forward(self, x):
        x = self.adjoint_conv_op(x)
        # input is CHW
        x2 = x[:, :int(x.shape[1]/2), :, :]
        x1 = x[:, int(x.shape[1]/2):, :, :]
        x1 = self.adjoint_up(x1)
        return (x1, x2)


class out_conv(nn.Module):
    def __init__(self, in_channels, out_channels, bias_bool = False):
        super(out_conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias_bool = bias_bool
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias= bias_bool)
    def forward(self, x):
        return self.conv(x)    
    

class adjoint_out_conv(nn.Module):
    def __init__(self, out_conv_model):
        super().__init__()
        assert out_conv_model.bias_bool == False
        in_channels = out_conv_model.out_channels
        out_channels = out_conv_model.in_channels

        self.adjoint_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, bias= False)
        self.adjoint_conv.weight = out_conv_model.conv.weight

    def forward(self, x):
        return self.adjoint_conv(x)
    
    
    
class dictionary_model(nn.Module):
    def __init__(self,  kernel_size, hidden_layer_width_list, n_classes, bilinear_bool = False, bias_bool = False):
        super(dictionary_model, self).__init__()
        
        self.hidden_layer_width_list = hidden_layer_width_list
        self.bilinear_bool = bilinear_bool
        self.bias_bool = bias_bool
        
        in_out_list = [ [hidden_layer_width_list[i], hidden_layer_width_list[i+1]] for i in  range(len(hidden_layer_width_list) -1) ]
        
        self.num_hidden_layers = len(in_out_list)
        
        self.n_classes = n_classes
        
        # the initial convolution on the bottleneck layer
        self.bottleneck_conv = nn.Conv2d(hidden_layer_width_list[0], hidden_layer_width_list[0], kernel_size=kernel_size, padding= kernel_size // 2, bias= self.bias_bool)

        self.syn_up_list = []

        for layer_idx in range(self.num_hidden_layers):
            new_up_block = up_block(kernel_size, *in_out_list[layer_idx], bilinear_bool = self.bilinear_bool, bias_bool = self.bias_bool)
            self.syn_up_list.append(new_up_block)           
        
        self.syn_up_list = nn.Sequential( *self.syn_up_list )
        
        self.syn_outc = out_conv(hidden_layer_width_list[-1], n_classes)

    def forward(self, x_list):
        # x_list is ordered from wide-channel to thin-channel.
        num_res_levels = len(x_list)
        
#         x_prev = x_list[0]
        
        x_prev = self.bottleneck_conv(x_list[0])
        
        for i in range(1, num_res_levels):
            x = x_list[i] 
            syn_up = self.syn_up_list[i-1]
            x_prev = syn_up(x_prev, x)
            
        syn_output = self.syn_outc(x_prev)
        return syn_output
    

class adjoint_dictionary_model(nn.Module):
    def __init__(self, dictionary_model):
        super().__init__()
        
        # to construct the adjoint model, one should exclude additive biases and use transposed conv for upsampling.
        assert dictionary_model.bias_bool == False
        assert dictionary_model.bilinear_bool == False
        
        self.adjoint_syn_outc = adjoint_out_conv(dictionary_model.syn_outc)
        self.adjoint_syn_bottleneck_conv = adjoint_conv_op(dictionary_model.bottleneck_conv)        

        self.adjoint_syn_up_list = []
        
        self.num_hidden_layers = dictionary_model.num_hidden_layers
        
        for layer_idx in range(dictionary_model.num_hidden_layers): 
            self.adjoint_syn_up_list.append(adjoint_up_block(dictionary_model.syn_up_list[layer_idx] ) )
            

    def forward(self, y):
        y = self.adjoint_syn_outc(y)
        x_list = []
        
        for layer_idx in range(self.num_hidden_layers-1, -1, -1):  
            adjoint_syn_up = self.adjoint_syn_up_list[layer_idx]
            y, x = adjoint_syn_up(y)
            x_list.append(x)
            
        y = self.adjoint_syn_bottleneck_conv(y)
        x_list.append(y)
        x_list.reverse()
        return x_list 

# # Here are some test cases that check whether the ajoint model is implemented correctly. 
# # from ista_unet.models import dictionary_model, adjoint_dictionary_model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # # generate some random codes at each resolution (do not have to be sparse)
# x1_input = torch.rand(2, 1024, 8, 8)
# x2_input = torch.rand(2, 512, 16, 16)
# x3_input = torch.rand(2, 256, 32, 32)
# x4_input = torch.rand(2, 128, 64, 64)
# x5_input = torch.rand(2, 64, 128, 128)

# x_list = [x1_input, x2_input, x3_input, x4_input, x5_input ]
# x_list = [x.to(device) for x in x_list]

# # instantiate a dictionary_model model
# model = dictionary_model(kernel_size = 3, hidden_layer_width_list = [1024, 512, 256, 128, 64], n_classes = 3).to(device)

# # output produced by the dictionary_model model
# model_out = model(x_list)

# # # randomly initialize a tensor of the same shape 
# rand_out = torch.rand(model_out.shape).to(device)

# # # compute the adjoint of that model
# adjoint_model = adjoint_dictionary_model(model)

# adjoint_model_out = adjoint_model(rand_out)

# # # The following two numbers should be really close. 
# print(torch.dot( model_out.flatten(), rand_out.flatten()) )
# print( sum([ torch.dot( x_list[i].flatten(),  adjoint_model_out[i].flatten() ) for i in range(len(x_list))  ]) )



class ista_steps(nn.Module):
    def __init__(self, dictionary_model, ista_num_steps, lasso_lambda_scalar, uncouple_dictionary_model = None):
        super().__init__()
        
        hidden_layer_width_list = dictionary_model.hidden_layer_width_list
        
        self.ista_num_steps = ista_num_steps
        self.dictionary_model = dictionary_model
        
        if uncouple_dictionary_model is None:
            self.adjoint_dictionary_model = adjoint_dictionary_model(dictionary_model)
        else:
            self.adjoint_dictionary_model = adjoint_dictionary_model(uncouple_dictionary_model)
            
        self.num_synthesis_layers = dictionary_model.num_hidden_layers + 1
        
        # calculate the dominant eigenvalue of the gram matrix
        with torch.no_grad():
            L = power_iteration_conv_model(self.dictionary_model, num_simulations = 20)     
            
        # initialize the step-size as a learnable parameter
        self.ista_stepsize = torch.nn.Parameter(1/L)
#         self.ista_stepsize.requires_grad = False

        # initialize a list of lasso lambda parameter. These params are initialized for each channel at each resolution level.
        lasso_lambda_list = [torch.nn.Parameter(lasso_lambda_scalar * torch.ones(1, width, 1, 1) ) for width in hidden_layer_width_list]                
    
    
        self.lasso_lambda_list =  torch.nn.ParameterList( lasso_lambda_list )


    
    def forward(self, y):
        
        ista_num_steps = self.ista_num_steps
        ista_stepsize =  self.ista_stepsize
        ista_stepsize.data =  relu(self.ista_stepsize, lambd = 0)       
        
        lasso_lambda_list =  self.lasso_lambda_list
                
        num_samples =  y.shape[0]
        
        x_list = initialize_sparse_codes(y, hidden_layer_width_list = self.dictionary_model.hidden_layer_width_list)
        x_list = [x.cuda() for x in  x_list]
        
        num_x = len(x_list)
        
        for idx in range(ista_num_steps):
            err = self.dictionary_model( x_list ) - y
            adj_err_list  = self.adjoint_dictionary_model(err)
            
            for i in range(num_x):
                # clamp each lasso_lambda to make sure that it is non-negative.
                lasso_lambda_list[i].data = relu(lasso_lambda_list[i].data, lambd = 0)
                
                # ista iteration
                x_list[i] = relu(x_list[i] - ista_stepsize * adj_err_list[i], lambd = ista_stepsize *  lasso_lambda_list[i] )
        return x_list
        
    
class ista_unet(nn.Module):
    def __init__(self, kernel_size, hidden_layer_width_list, n_classes, ista_num_steps, lasso_lambda_scalar, uncouple_adjoint_bool = False, tied_bool = False, relu_out_bool = False, bilinear_up_bool = False, bias_up_bool = False, **kwargs):

        super(ista_unet, self).__init__()
        self.n_classes = n_classes
        
        self.analysis_model = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)

        # Configure the dictionary for the inputs
        if uncouple_adjoint_bool:
            self.uncouple_dictionary_model = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        else:
            self.uncouple_dictionary_model = None
            
        self.sparse_coder = ista_steps(self.analysis_model, ista_num_steps, lasso_lambda_scalar, uncouple_dictionary_model = self.uncouple_dictionary_model)
        
        
        # Configure the dictionary for the targets
        if tied_bool:
            self.synthesis_model = self.analysis_model
        else:
            self.synthesis_model = dictionary_model(kernel_size, hidden_layer_width_list, n_classes, bilinear_bool = bilinear_up_bool, bias_bool = bias_up_bool)
               

#         if use_sigmoid:
#             self.nonlin = nn.Sigmoid()
#         else:
#             self.nonlin = nn.Identity()

        if relu_out_bool:
            self.nonlin = nn.ReLU()
        else:
            self.nonlin = nn.Identity()

    def forward(self, x):
        x_list = self.sparse_coder(x)
        output = self.synthesis_model(x_list) 
        output = self.nonlin(output)
        return output