-- @Author: blcv
-- @Date:   2015-12-10 08:28:59
-- @Last Modified by:   blcv
-- @Last Modified time: 2015-12-10 09:39:20
require 'nn'
require 'nngraph'

local QSAMPLER = {}

function QSAMPLER.create(rnn_size, next_h, size_z_layer)
  -- get two parameters from net
  mu = nn.Linear(rnn_size, size_z_layer)(next_h)
  sigma = nn.Linear(rnn_size, size_z_layer)(next_h)
  --calculate z state
  sigma = nn.Exp()(sigma)
  e = nn.Identity()()
  sigma_e = nn.CMulTable()({sigma, e})
  z = nn.CAddTable()({mu, sigma_e})
  --calculate loss_z
  mu_squared = nn.Square()(mu)
  sigma_squared = nn.Square()(sigma)
  log_sigma_sq = nn.Log()(sigma_squared)
  minus_log_sigma = nn.MulConstant(-1)(log_sigma_sq)
  loss_z = nn.CAddTable()({mu_squared, sigma_squared, minus_log_sigma})
  loss_z = nn.AddConstant(-1)(loss_z)
  loss_z = nn.MulConstant(0.5)(loss_z)
  loss_z = nn.Sum(2)(loss_z)
  return nn.gModule({next_h,e}, {z, loss_z})

end
return QSAMPLER

