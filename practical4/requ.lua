require 'nn'

local ReQU = torch.class('nn.ReQU', 'nn.Module')

function ReQU:updateOutput(input)
  self.output = torch.Tensor()
  self.output:resizeAs(input):copy(input)
  self.output:cmul(torch.gt(input,0):double()):cmul(input)
  return self.output
end

function ReQU:updateGradInput(input, gradOutput)
  self.gradInput = torch.Tensor()
  self.gradInput:resizeAs(input):copy(input)
  self.gradInput:cmul(torch.gt(input, 0):double()):cmul(gradOutput):mul(2)
  return self.gradInput
end

