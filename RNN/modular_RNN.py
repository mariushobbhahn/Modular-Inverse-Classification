import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils import plot_sequence


class ModularRNN(nn.Module):

    def __init__(self, list_of_networks, loss_function=nn.MSELoss()):
        super(ModularRNN, self).__init__()
        self.list_of_networks = list_of_networks

        #get sequence to start generative model
        zeros = torch.zeros(205) #change this for other sequence lengths
        zeros[0] = 1
        input_sequence = zeros.view((205, 1))
        self.input = input_sequence

        self.loss_function = loss_function


    def inverse_classification_rnn(self, gen_net, target, iterations, plot_results=False):
        """
        takes a target sequence as input and returns the difference between
        first layer activations after backpropagating the target.

        :param
            gen_net: generative network
            target: sequence
            iterations: number of iterations of backprops
        :return:
            difference between activations and gradient
        """

        params = list(gen_net.parameters())
        default_params  = params[0]
        #print("default params: ", default_params)

        for i in range(iterations):
            #get the prediction of the generative network:
            out = gen_net(self.input)
            loss = self.loss_function(out, target)

            #propagate the loss back
            loss.backward()

            #print("inputs to lstm: ", gen_net.lstm.weight_ih_l0)
            #print("inputs to lstm chunked: ", gen_net.lstm.weight_ih_l0.chunk(4, 0))
            params = list(gen_net.parameters())
            i_grads = params[0].grad.chunk(4, 0)[0]
            #print("gradients to lstm chunked: ", i_grads)

            #change inputs (does have no effect)
            #self.input[0] = self.input[0] + torch.sum(i_grads)


            #change linear inpur parameters before LSTM temporarily
            lr = 0.1
            current_params = params[0]
            updated_params = current_params + torch.cat((lr*i_grads, torch.zeros(30,1)), 0)
            gen_net.state_dict()['lstm.weight_ih_l0'].data.copy_(updated_params)

            #access gradients
            #for p in gen_net.named_parameters():
            #    print("p: ", p)
            # for p in gen_net.parameters():
            #     print("parameter sizes: ", p.size())
            #     print("gradient sizes: ", p.grad.size())
            #     print("name of the parameter: ", p.name)

        if plot_results:
            plot_sequence(gen_net(self.input).detach().numpy(), swapaxis=True, title='prediction after gradient iteration')
        diff = self.loss_function(gen_net(self.input), target)

        return(diff)

    def inverse_classification_all(self, target, iterations):

        loss_list = []
        #calculate loss between target and generative model for all networks
        for net in self.list_of_networks:
            loss_net = self.inverse_classification_rnn(gen_net=net, target=target, iterations=iterations)
            loss_list.append(loss_net)

        print("loss_list: ", loss_list)

        return(loss_list)






