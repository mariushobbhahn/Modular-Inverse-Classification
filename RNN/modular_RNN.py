import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils import plot_sequence
from torch.distributions import uniform
import numpy as np

class ModularRNN(nn.Module):

    def __init__(self, list_of_networks, loss_function=nn.MSELoss(), input_dim=4):
        super(ModularRNN, self).__init__()
        self.list_of_networks = list_of_networks
        self.input_dim = input_dim


        self.loss_function = loss_function


    def inverse_classification_rnn(self, gen_net, target, iterations, plot_results=True):
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

        def adam_optimizer_step(input_vector, gradient_vector, m_t1=0, v_t1=0, beta1=0.9, beta2=0.99, eta=0.0001,
                           epsilon=1e-8):
            # get moment terms
            m_t = m_t1 * beta1 + (1 - beta1) * gradient_vector
            v_t = v_t1 * beta2 + (1 - beta2) * np.square(gradient_vector)
            m_t = torch.Tensor(m_t)
            v_t = torch.Tensor(v_t)

            epsilon = np.full(shape=np.shape(gradient_vector), fill_value=epsilon)
            epsilon = torch.Tensor(epsilon)
            dx = (eta * m_t) / (np.sqrt(v_t) + epsilon)
            dx = torch.Tensor(dx)
            output_vector = input_vector - dx

            return (output_vector, m_t, v_t)

        params = list(gen_net.parameters())
        default_params  = params[0]
        #print("default params: ", default_params)

        # get sequence to start generative model
        zero_sequence = torch.zeros((205, self.input_dim))
        input_sequence = zero_sequence
        # uniform_vector = torch.tensor([1/input_dim] * input_dim)
        # input_sequence[0] = uniform_vector
        m = uniform.Uniform(0, 0.5)
        uniform_sample = m.sample((self.input_dim,))
        print("new uniform sample: ", uniform_sample)
        input_sequence[0] = uniform_sample
        self.input = input_sequence

        m_t1 = 0
        v_t1 = 0

        for i in range(iterations):

            #clip and normalize input
            self.input[0] = self.input[0].clamp(min=0, max=1)
            self.input[0] = self.input[0] / torch.sum(self.input[0])
            self.input = torch.autograd.Variable(self.input, requires_grad=True)
            print("self.input: ", self.input[0])


            #get the prediction of the generative network:
            out = gen_net(self.input)
            #plot_sequence(target.detach().numpy(), "target", swapaxis=True)
            #plot_sequence(out.detach().numpy(), "test", swapaxis=True)
            loss = self.loss_function(out, target)
            print('loss: ', loss)

            #propagate the loss back
            loss.backward()

            input_gradient = self.input.grad[0]
            print("input_gradient: ", input_gradient)

            #print("inputs to lstm: ", gen_net.lstm.weight_ih_l0)
            #print("inputs to lstm chunked: ", gen_net.lstm.weight_ih_l0.chunk(4, 0))
            params = list(gen_net.parameters())
            #for param in params:
            #    print("param: ", param.size())
            #i_grads = params[0].grad.chunk(4, 0)[0]
            #print("params[0].grad(): ", params[0].grad)
            #print("gradients to lstm chunked: ", i_grads.size())

            #""" #this is the version where the inputs are changed not the gradients
            #print("i_grads: ", i_grads)
            #change_values = torch.sum(i_grads, dim=0)
            #print("change_values: ", change_values)

            #add it on the input
            lr = 0.01
            #self.input[0] = self.input[0] + lr * input_gradient
            self.input[0], m_t1, v_t1 = adam_optimizer_step(input_vector=self.input[0], gradient_vector=input_gradient,
                                                        eta=lr, m_t1=m_t1, v_t1=v_t1)
            #print("self.input after addition: ", self.input[0])
            #"""

            """ #this is the version where we update the parameters directly instead of adding it on the inputs
            #change linear inpur parameters before LSTM temporarily
            lr = 1
            current_params = params[0]
            updated_params = current_params + torch.cat((lr*i_grads, torch.zeros(30,4)), 0)
            gen_net.state_dict()['lstm.weight_ih_l0'].data.copy_(updated_params)
            #"""

            #access gradients
            #for p in gen_net.named_parameters():
            #    print("p: ", p)
            # for p in gen_net.parameters():
            #     print("parameter sizes: ", p.size())
            #     print("gradient sizes: ", p.grad.size())
            #     print("name of the parameter: ", p.name)

        if plot_results:
            print("plot result")
            plot_sequence(gen_net(self.input).detach().numpy(), swapaxis=True, title='prediction after gradient iteration')
            print("plot target")
            plot_sequence(target.detach().numpy(), swapaxis=True, title='target')
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






