require 'nn'
require 'nngraph'

function duplicate(x, size_image)
  local y = nn.Reshape(1)(x)
  local l = {}
  for i = 1, size_image do 
    l[#l + 1] = nn.Copy()(y)  
  end
  local z = nn.JoinTable(2)(l)
  return z
end

function createParameters(input, rnn_size, size_image, size_attention)
  gx = duplicate(nn.Linear(rnn_size, 1)(input), size_image)
  gx = duplicate(nn.Linear(rnn_size, 1)(input), size_image)
  gy = duplicate(nn.Linear(rnn_size, 1)(input), size_image)
  delta = duplicate(nn.Linear(rnn_size, 1)(input), size_image)
  gamma = duplicate(nn.Linear(rnn_size, 1)(input), size_image)
  sigma = duplicate(nn.Linear(rnn_size, 1)(input), size_image)
  delta = nn.Exp()(delta)
  gamma = nn.Exp()(gamma)
  sigma = nn.Exp()(sigma)
  sigma = nn.Power(-2)(sigma)
  sigma = nn.MulConstant(-1/2)(sigma)
  gx = nn.AddConstant(1)(gx)
  gy = nn.AddConstant(1)(gy)
  gx = nn.MulConstant((size_image + 1) / 2)(gx)
  gy = nn.MulConstant((size_image + 1) / 2)(gy)
  delta = nn.MulConstant((math.max(size_image,size_image)-1)/(size_attention-1))(delta)
end

function genrFilters(g, ascending, size_image, size_attention, sigma, gamma, batchSize)
  filters = {}
  for i = 1, size_attention do
      mu_i = nn.CAddTable()({g, nn.MulConstant(i - size_attention/2 - 1/2)(delta)})
      mu_i = nn.MulConstant(-1)(mu_i)
      d_i = nn.CAddTable()({mu_i, ascending})
      d_i = nn.Power(2)(d_i)
      exp_i = nn.CMulTable()({d_i, sigma})
      exp_i = nn.Exp()(exp_i)
      exp_i = nn.View(batchSize, 1, size_image)(exp_i)
      filters[#filters + 1] = nn.CMulTable()({exp_i, gamma})
  end
  filterbank = nn.JoinTable(2)(filters)
  --TODO: normalize filter to Sum[filterbank]=1
--   filterbank = nn.Sum(1,)
  return filterbank
end

function applyFilter(x, filterbank_x, filterbank_y)
  local patch = nn.MM()({filterbank_y, x})
  local patch = nn.MM(false, true)({patch, filterbank_x})
  return patch
end
 
