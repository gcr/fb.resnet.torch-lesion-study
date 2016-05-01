-- Code from https://github.com/yueatsprograms/Stochastic_Depth/blob/master/StochasticDrop.lua

require 'nn'
require 'cudnn'
require 'cunn'

local StochasticDrop, parent = torch.class('nn.StochasticDrop', 'nn.Module')

function StochasticDrop:__init(deathRate)
    parent.__init(self)
    self.gradInput = torch.Tensor()
    self.gate = true
    self.train = true
    self.deathRate = deathRate or 0.5
end

function StochasticDrop:updateOutput(input)
    -- local skip_forward = self.skip:forward(input)
    self.output:resizeAs(input):copy(input)
    --print(self.deathRate)
    self.gate = (torch.rand(1)[1] > self.deathRate)
    if self.train then
      if not self.gate then -- only compute convolutional output when gate is open
         self.output:mul(0.0) --add(self.net:forward(input))
      end
    else
      self.output:mul(1-self.deathRate)
    end
    return self.output
end

function StochasticDrop:updateGradInput(input, gradOutput)
   self.gradInput = self.gradInput or input.new()
   self.gradInput:resizeAs(input):copy(gradOutput)
   if not self.gate then
      self.gradInput:mul(0) -- add(self.net:updateGradInput(input, gradOutput))
   end
   return self.gradInput
end

function StochasticDrop:accGradParameters(input, gradOutput, scale)
   -- scale = scale or 1
   -- if self.gate then
   --    self.net:accGradParameters(input, gradOutput, scale)
   -- end
end
