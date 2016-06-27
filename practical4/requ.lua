require 'nn'

local ReQU = torch.class('nn.ReQU', 'nn.Module')

function ReQU:updateOutput(input)
  -- TODO
  self.output:resizeAs(input):copy(input)
  -- ...something here...
  self.output:apply(function(x) if (x>0) then x = x^2 else x = 0 end; return x end)
  return self.output
end

function ReQU:updateGradInput(input, gradOutput)
  -- TODO
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  -- ...something here...
  return self.gradInput
end

