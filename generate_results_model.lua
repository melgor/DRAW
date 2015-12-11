-- @Author: blcv
-- @Date:   2015-12-09 16:14:38
-- @Last Modified by:   blcv
-- @Last Modified time: 2015-12-11 10:34:24
require 'nn'
require 'cunn'
require 'nngraph'
require 'optim'
require 'image'
require 'socket'

local model_utils=require 'model_utils'
local mnist = require 'mnist'
torch.setdefaulttensortype('torch.CudaTensor')
--STH does not work, the output are rather messy. Molder have been learning by 5 hours, then it stop to converge

cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training a DRAW on MNIST')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Model And Training Regime')
cmd:option('-model',       './Results/dec.net',             'Decoder model to use')
cmd:option('-batchSize',          256,                      'batch size')
cmd:option('-type',               'cuda',                   'float or cuda')
cmd:option('-devid',              1,                        'device ID (if using CUDA)')
cmd:option('-seed',               123,                      'torch manual random number generator seed')
cmd:option('-save',               "Generation/",                   'folder to save generation')

cmd:text('===>DRAW Option')
cmd:option('-sizeImage',            28,                     'size of image to use, for MNIST 28')
cmd:option('-sizeLayerZ',           20,                     'size of last layer of encoder, which store information')
cmd:option('-rnnSize',             100,                     'size of hidden layer in RNN')
cmd:option('-seqSize',              50,                     'number of sequences in RNN')


opt = cmd:parse(arg or {})
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')
if opt.type == 'cuda' then
    cutorch.setDevice(opt.devid)
    cutorch.manualSeed(opt.seed)
    torch.setdefaulttensortype('torch.CudaTensor')
end
--load model
decoder = torch.load(opt.model)

--generation
local lstm_c_dec = {[0]=torch.zeros(opt.batchSize, opt.rnnSize)}
local lstm_h_dec = {[0]=torch.zeros(opt.batchSize, opt.rnnSize)}
local canvas = {[0]=torch.zeros(opt.batchSize, opt.sizeImage, opt.sizeImage)}
local z = {}
local x = {}
local x_error = {}
local x_prediction = {}
local loss_x = {}
local ascending = torch.zeros(opt.batchSize, opt.sizeImage)
for k = 1, opt.batchSize do
  for i = 1, opt.sizeImage do 
      ascending[k][i] = i
  end
end

local sigmoid = nn.Sigmoid() 
for t = 1, opt.seqSize do
      z[t] = torch.randn(opt.batchSize, opt.sizeLayerZ)
      x[t] = torch.rand(opt.batchSize, opt.sizeImage, opt.sizeImage)
      lstm_c_dec[t], lstm_h_dec[t]          = unpack(decoder[1]:forward({z[t],lstm_c_dec[t-1], lstm_h_dec[t-1]}))
      canvas[t]                             = decoder[2]:forward({lstm_h_dec[t],ascending,canvas[t-1]})
      loss_x[t],x_prediction[t],x_error[t]  = unpack(decoder[3]:forward({canvas[t],x[t]}))

      torch.setdefaulttensortype('torch.FloatTensor')
      x_prediction[t] = x_prediction[t]:float()
      local display = image.toDisplayTensor(x_prediction[t])
      image.save(opt.save .. "glimpse-output" .. socket.gettime()*10000 ..".png", display)
      torch.setdefaulttensortype('torch.CudaTensor')
end

os.execute(string.format('convert -delay 10 %s/glimpse-output*   %s/sequence.gif',opt.save,opt.save))

-- torch.setdefaulttensortype('torch.FloatTensor')

-- for t = 0, opt.seqSize do  
--   canvas[t] = canvas[t]:float()
--   local x_pred = sigmoid:forward(canvas[t])
--   local  x_pred = x_pred:float()
--   -- graph.dot(decoder.fg, 'MLP', 'mlp')
--   -- print(x_pred:gt(0.5))
--   local display = image.toDisplayTensor(x_pred)
--   image.save(opt.save .. t .. "_glimpse-output.png", display)
-- end


-- image.display(display)
-- x_pred = x_prediction[50][1]:gt(0.5):float()


-- torch.setdefaulttensortype('torch.FloatTensor')
-- for t = 1, seq_length do
--   x_pred = x_prediction[t]:float():gt(0.5)
--   display = image.toDisplayTensor(x_pred)
--   image.save(t .."glimpse-output.png", display)
-- end
-- image.display(display)
-- torch.save('x_generation', x_prediction)

-- x_prediction = torch.load('x_prediction')

-- x = torch.zeros(#x_prediction, x_prediction[1]:size(2), x_prediction[1]:size(3)) 
-- for i = 1, x_prediction[1]:size(1) do
--   for t = 1, #x_prediction do 
--     local data = x_prediction[t][i]:gt(0.5)
--     print(#x[{{t}, {}, {}}])
--     -- x[{{t}, {}, {}}] = data
--     image.display(data)
--   end
--   image.display(x)
-- end
