-- @Author: blcv
-- @Date:   2015-12-11 09:10:03
-- @Last Modified by:   blcv
-- @Last Modified time: 2015-12-11 10:33:39
require 'nn'
require 'cunn'
require 'nngraph'
require 'optim'
require 'image'
require 'socket'

local model_utils=require 'model_utils'
local mnist = require 'mnist'
torch.setdefaulttensortype('torch.CudaTensor')


cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training a DRAW on MNIST')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Model And Training Regime')
cmd:option('-modelEnc',       './Results/enc.net',             'Decoder model to use')
cmd:option('-modelDec',       './Results/dec.net',             'Decoder model to use')
cmd:option('-batchSize',          256,                      'batch size')
cmd:option('-type',               'cuda',                   'float or cuda')
cmd:option('-devid',              1,                        'device ID (if using CUDA)')
cmd:option('-seed',               123,                      'torch manual random number generator seed')
cmd:option('-save',               "Reconstruct/",                   'folder to save generation')

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
encoder = torch.load(opt.modelEnc)
decoder = torch.load(opt.modelDec)

  
      
--train
local trainset = mnist.traindataset()
local features_input = torch.zeros(opt.batchSize, opt.sizeImage, opt.sizeImage)
for i = 1, opt.batchSize do
    features_input[{{i}, {}, {}}] = trainset[i].x:gt(125):cuda() 
end

  

--generation
local lstm_c_enc = {[0]=torch.zeros(opt.batchSize, opt.rnnSize)}
local lstm_h_enc = {[0]=torch.zeros(opt.batchSize, opt.rnnSize)}
local lstm_c_dec = {[0]=torch.zeros(opt.batchSize, opt.rnnSize)}
local lstm_h_dec = {[0]=torch.zeros(opt.batchSize, opt.rnnSize)}
local x_error = {[0]=torch.rand(opt.batchSize, opt.sizeImage, opt.sizeImage)}
local x_prediction = {}
local loss_x = {}
local loss_z = {}
local canvas = {[0]=torch.zeros(opt.batchSize, opt.sizeImage, opt.sizeImage)}
local x = {}
local z = {}
local patch = {}
local read_input = {}
local e = {}
local ascending = torch.zeros(opt.batchSize, opt.sizeImage)

for k = 1, opt.batchSize do
  for i = 1, opt.sizeImage do 
      ascending[k][i] = i
  end
end

print(decoder[2]) 
timer = torch.Timer()
for t = 1, opt.seqSize do
      -- x[t] = torch.rand(opt.batchSize, opt.sizeImage, opt.sizeImage)
      x[t] = features_input
      e[t] = torch.randn(opt.batchSize, opt.sizeLayerZ)
      z[t] = torch.randn(opt.batchSize, opt.sizeLayerZ)
      --encoder
      patch[t], read_input[t]         = unpack(encoder[1]:forward({x[t], x_error[t-1], lstm_h_dec[t-1], ascending}))
      lstm_c_enc[t], lstm_h_enc[t]    = unpack(encoder[2]:forward({read_input[t],lstm_c_enc[t-1], lstm_h_enc[t-1]}))
      z[t], loss_z[t]                 = unpack(encoder[3]:forward({lstm_h_enc[t], e[t]}))
      --decoder    
      lstm_c_dec[t], lstm_h_dec[t]          = unpack(decoder[1]:forward({z[t],lstm_c_dec[t-1], lstm_h_dec[t-1]}))
      canvas[t]                             = decoder[2]:forward({lstm_h_dec[t],ascending,canvas[t-1]})
      loss_x[t],x_prediction[t],x_error[t]  = unpack(decoder[3]:forward({canvas[t],x[t]}))
      torch.setdefaulttensortype('torch.FloatTensor')
      x_prediction[t] = x_prediction[t]:float()
      local display = image.toDisplayTensor(x_prediction[t])
      image.save(opt.save .. "glimpse-output" .. socket.gettime()*10000 ..".png", display)
      torch.setdefaulttensortype('torch.CudaTensor')
end

--save input
torch.setdefaulttensortype('torch.FloatTensor')
features_input  = features_input:float()
local display = image.toDisplayTensor(features_input)
image.save(opt.save .."glimpse-input.png", display)


os.execute(string.format('convert -delay 10 %s/glimpse-output*   %s/sequence.gif',opt.save,opt.save))

-- function print_outputs(m)
--   print(m)
--   print(#m.output)
-- end
-- decoder[2]:apply(print_outputs)