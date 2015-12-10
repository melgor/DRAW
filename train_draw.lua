-- require 'mobdebug'.start()

require 'nn'
require 'cunn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'
local LSTM = require 'draw_nngraph.lstm'
local READ = require 'draw_nngraph.read'
local WRITE = require 'draw_nngraph.write'
local QSampler = require 'draw_nngraph.qsampler'
local LOSS_X = require 'draw_nngraph.loss_x'
local c = require 'trepl.colorize'
-- nngraph.setDebug(true)


cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training a DRAW on MNIST')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Model And Training Regime')
cmd:option('-saveFolder',       './Results/',               'folder to save')
cmd:option('-LR',                 0.01,                     'learning rate')
cmd:option('-LRDecay',            0,                        'learning rate decay (in # samples)')
cmd:option('-weightDecay',        5e-4,                     'L2 penalty on the weights')
cmd:option('-momentum',           0.9,                      'momentum')
cmd:option('-batchSize',          256,                      'batch size')
cmd:option('-optimization',       'sgd',                    'optimization method')
cmd:option('-seed',               123,                      'torch manual random number generator seed')
cmd:option('-epoch',              10,                       'number of epochs to train, -1 for unbounded')
cmd:option('-clipGradient',       5,                       'gradient clipping value for rnn')

cmd:text('===>Platform Optimization')
cmd:option('-threads',            8,                        'number of threads')
-- cmd:option('-type',               'cuda',                   'float or cuda')
cmd:option('-devid',              1,                        'device ID (if using CUDA)')


cmd:text('===>DRAW Option')
cmd:option('-sizeImage',            28,                     'size of image to use, for MNIST 28')
cmd:option('-sizeLayerZ',           20,                     'size of last layer of encoder, which store information')
cmd:option('-rnnSize',             100,                     'size of hidden layer in RNN')
cmd:option('-seqSize',              50,                     'number of sequences in RNN')
cmd:option('-attenReadSize',         3,                     'size of glimpse which read data')
cmd:option('-attenWriteSize',        3,                     'size of glimpse which write data')

opt = cmd:parse(arg or {})
print(opt)
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')
if opt.type == 'cuda' then
    cutorch.setDevice(opt.devid)
    cutorch.manualSeed(opt.seed)
    torch.setdefaulttensortype('torch.CudaTensor')
end


--encoder 
x = nn.Identity()()
x_error_prev = nn.Identity()()
-- read operator from image
read_module = READ.create(x, x_error_prev, opt.rnnSize, opt.sizeImage, opt.attenReadSize, opt.batchSize)
-- input = read_input
-- RNN layer which take input and previous state
input = nn.Identity()()
lstm_enc = LSTM.create(input, 2 * opt.attenReadSize * opt.attenReadSize, opt.rnnSize)

--QSampler
next_h = nn.Identity()()
qsampler = QSampler.create(opt.rnnSize, next_h, opt.sizeLayerZ)
--combine everything into encoder
if opt.type == 'cuda' then 
  encoder = {read_module:cuda(), lstm_enc:cuda(), qsampler:cuda()}
else
  encoder = {read_module, lstm_enc, qsampler}
end
encoder.name = 'encoder'

--decoder
input = nn.Identity()()
lstn_dec = LSTM.create(input, opt.sizeLayerZ, opt.rnnSize)

next_h = nn.Identity()()
prev_canvas = nn.Identity()()
write_module = WRITE.create(next_h, prev_canvas, opt.rnnSize, opt.sizeImage, opt.attenWriteSize, opt.batchSize)

x = nn.Identity()()
next_canvas = nn.Identity()()
loss_x = LOSS_X.create(x,next_canvas)
if opt.type == 'cuda' then
  decoder = {lstn_dec:cuda(), write_module:cuda(), loss_x:cuda()}
else
  decoder = {lstn_dec, write_module, loss_x}
end
decoder.name = 'decoder'



print ("Model build")



--train
trainset = mnist.traindataset()
-- testset = mnist.testdataset()

params, grad_params = model_utils.combine_all_parameters(encoder[1], encoder[2], encoder[3], decoder[1], decoder[2], decoder[3])
encoder_clones = model_utils.clone_many_times_multiple_nngraph(encoder, opt.seqSize)
decoder_clones = model_utils.clone_many_times_multiple_nngraph(decoder, opt.seqSize)

ascending = torch.zeros(opt.batchSize, opt.sizeImage)
for k = 1, opt.batchSize do
  for i = 1, opt.sizeImage do 
      ascending[k][i] = i
  end
end

------------------------------------------------------------------------
-- optimization loop
--
optim_state = {learningRate = opt.LR}
epoch = 0
while epoch ~= opt.epoch do
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  local targets = torch.zeros(opt.batchSize)
  torch.setdefaulttensortype('torch.FloatTensor')
  local indices = torch.randperm(trainset.size):long():split(opt.batchSize)
  -- local indices = torch.randperm(opt.batchSize):long():split(opt.batchSize)
  torch.setdefaulttensortype('torch.CudaTensor')
  local tic = torch.tic()
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)
    inputs = torch.zeros(opt.batchSize, opt.sizeImage, opt.sizeImage)
    for i = 1, v:size(1) do
      inputs[{{i}, {}, {}}] = trainset[v[i]].x:gt(125):cuda()
      targets[i] =  trainset[v[i]].y
    end

        -- do fwd/bwd and return loss, grad_params
    function feval(x_arg)
        if x_arg ~= params then
            params:copy(x_arg)
        end
        grad_params:zero()
        
        ------------------- forward pass -------------------
        lstm_c_enc = {[0]=torch.zeros(opt.batchSize, opt.rnnSize)}
        lstm_h_enc = {[0]=torch.zeros(opt.batchSize, opt.rnnSize)}
        lstm_c_dec = {[0]=torch.zeros(opt.batchSize, opt.rnnSize)}
        lstm_h_dec = {[0]=torch.zeros(opt.batchSize, opt.rnnSize)}
        x_error = {[0]=torch.zeros(opt.batchSize, opt.sizeImage, opt.sizeImage)}
        x_prediction = {}
        loss_z = {}
        loss_x = {}
        canvas = {[0]=torch.zeros(opt.batchSize, opt.sizeImage, opt.sizeImage)}
        x = {}
        patch = {}
        read_input = {}
        
        
        local loss = 0

        for t = 1, opt.seqSize do
          e[t] = torch.randn(opt.batchSize, opt.sizeLayerZ)
          x[t] = inputs
          --encoder
          patch[t], read_input[t]         = unpack(encoder_clones[t][1]:forward({x[t], x_error[t-1], lstm_h_dec[t-1], ascending}))
          lstm_c_enc[t], lstm_h_enc[t]    = unpack(encoder_clones[t][2]:forward({read_input[t],lstm_c_enc[t-1], lstm_h_enc[t-1]}))
          z[t], loss_z[t]                 = unpack(encoder_clones[t][3]:forward({lstm_h_enc[t], e[t]}))
          --decoder
          lstm_c_dec[t], lstm_h_dec[t]          = unpack(decoder_clones[t][1]:forward({z[t],lstm_c_dec[t-1], lstm_h_dec[t-1]}))
          canvas[t]                             = decoder_clones[t][2]:forward({lstm_h_dec[t],ascending,canvas[t-1]})
          loss_x[t],x_prediction[t],x_error[t]  = unpack(decoder_clones[t][3]:forward({canvas[t],x[t]}))
          
          loss = loss + torch.mean(loss_z[t]) + torch.mean(loss_x[t])
        end
        loss = loss / opt.seqSize

        ------------------ backward pass -------------------
        -- complete reverse order of the above
        dlstm_c_enc = {[opt.seqSize] = torch.zeros(opt.batchSize, opt.rnnSize)}
        dlstm_h_enc = {[opt.seqSize] = torch.zeros(opt.batchSize, opt.rnnSize)}
        dlstm_c_dec = {[opt.seqSize] = torch.zeros(opt.batchSize, opt.rnnSize)}
        dlstm_h_dec = {[opt.seqSize] = torch.zeros(opt.batchSize, opt.rnnSize)}
        dlstm_h_dec1 = {[opt.seqSize] = torch.zeros(opt.batchSize, opt.rnnSize)}
        dlstm_h_dec2 = {[opt.seqSize] = torch.zeros(opt.batchSize, opt.rnnSize)}
        dlstm_h_dec3 = {[opt.seqSize] = torch.zeros(opt.batchSize, opt.rnnSize)}
        dx_error = {[opt.seqSize] = torch.zeros(opt.batchSize, opt.sizeImage, opt.sizeImage)}
        dx_prediction = {}
        dloss_z = {}
        dloss_x = {}
        dcanvas1 = {[opt.seqSize] = torch.zeros(opt.batchSize, opt.sizeImage, opt.sizeImage)}
        dcanvas2 = {[opt.seqSize] = torch.zeros(opt.batchSize, opt.sizeImage, opt.sizeImage)}
        dz = {}
        dx1 = {}
        dx2 = {}
        de = {}
        dpatch = {}
        dread_input = {}
        
        for t = opt.seqSize,1,-1 do
          dloss_x[t] = torch.ones(opt.batchSize, 1)
          dloss_z[t] = torch.ones(opt.batchSize, 1)
          dx_prediction[t] = torch.zeros(opt.batchSize, opt.sizeImage, opt.sizeImage)
          dpatch[t] = torch.zeros(opt.batchSize, opt.attenWriteSize, opt.attenWriteSize) --TODO: Not sure if should be Read or Write Size

          --decoder
          dcanvas2[t],dx1[t]                          = unpack(decoder_clones[t][3]:backward({canvas[t],x[t]},{dloss_x[t],dx_prediction[t],dx_error[t]}))
          --merge gradient from canvas
          dcanvas1[t] = dcanvas1[t] + dcanvas2[t]
          dlstm_h_dec3[t],dascending1,dcanvas1[t-1]   = unpack(decoder_clones[t][2]:backward({lstm_h_dec[t],ascending,canvas[t-1]},dcanvas1[t]))
          --merge gradient from lstm_h_dec1
          dlstm_h_dec1[t] = (dlstm_h_dec1[t] + dlstm_h_dec3[t])
          dz[t], dlstm_c_dec[t-1], dlstm_h_dec1[t-1]  = unpack(decoder_clones[t][1]:backward({z[t],lstm_c_dec[t-1], lstm_h_dec[t-1]},{dlstm_c_dec[t],dlstm_h_dec1[t] }))

          --encoder
          dlstm_h_enc[t], de[t]                                 = unpack(encoder_clones[t][3]:backward({lstm_h_enc[t], e[t]},{dz[t],dloss_z[t]}))
          dread_input[t], dlstm_c_enc[t-1], dlstm_h_enc[t-1]    = unpack(encoder_clones[t][2]:backward({read_input[t],lstm_c_enc[t-1], lstm_h_enc[t-1]},{dlstm_c_enc[t], dlstm_h_enc[t]}))
          dx2[t], dx_error[t-1], dlstm_h_dec2[t-1], dascending2 = unpack(encoder_clones[t][1]:backward({x[t], x_error[t-1], lstm_h_dec[t-1], ascending},{dpatch[t],dread_input[t]}))
          --merge gradient from lstm_h_dec
          dlstm_h_dec[t-1] = dlstm_h_dec1[t-1] + dlstm_h_dec2[t-1]
        end

        -- clip gradient element-wise
        grad_params:clamp(-opt.clipGradient, opt.clipGradient)
        return loss, grad_params
    end
    _, loss = optim.adagrad(feval, params, optim_state)
  end
  print(string.format("epoch %4d, loss = %6.6f, time: %.2f s'", epoch, loss[1],  torch.toc(tic)))
  epoch = epoch + 1
  local filename_enc = paths.concat(opt.saveFolder, 'enc.net')
  local filename_dec = paths.concat(opt.saveFolder, 'dec.net')
  print('==> saving model to '..filename_enc)
  torch.save(filename_enc, encoder)
  torch.save(filename_dec, decoder)
end


-- --к чему стремимся
-- print(x[1][1]:gt(0.5))

-- --что получаем со временем
-- for t = 1, opt.seqSize do
--   print(patch[t][1]:gt(0.5))
--   print(x_prediction[t][1]:gt(0.5))
-- end


-- torch.save('x_prediction', x_prediction)


-- --generation
-- for t = 1, opt.seqSize do
--       e[t] = torch.randn(opt.batchSize, opt.sizeLayerZ)
--       x[t] = features_input
--       z[t] = torch.randn(opt.batchSize, opt.sizeLayerZ)
--       x_prediction[t], x_error[t], lstm_c_dec[t], lstm_h_dec[t], canvas[t], loss_x[t] = unpack(decoder_clones[t]:forward({x[t], z[t], lstm_c_dec[t-1], lstm_h_dec[t-1], canvas[t-1], ascending}))
--   end

-- torch.save('x_generation', x_prediction)

