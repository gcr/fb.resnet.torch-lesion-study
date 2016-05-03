--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'Stochastic_Depth/ResidualDrop'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'
local msgpack = require 'MessagePack'

torch.setdefaulttensortype('torch.FloatTensor')

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load model
local model = torch.load(opt.modelPath)
model:evaluate()
local criterion = nn.CrossEntropyCriterion():cuda()

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- Seek and destroy: Gather all resnet modules.
function gather_residual_blocks(model)
    -- A residual block is a block that has type 'nn.Sequential' and ends with
    -- an 'nn.CAddTable'. (This is true for the 200-layer module)
    local found_resnet_blocks = {}
    model:apply(function(module)
        if (torch.typename(module)=='nn.Sequential'
            and module.modules ) then
            if (torch.typename(module.modules[#module.modules-1])=='nn.CAddTable'
                or torch.typename(module.modules[#module.modules])=='nn.CAddTable') then
                found_resnet_blocks[#found_resnet_blocks+1] = module
            end
        end
        if (torch.typename(module)=='nn.ResidualDrop') then
           -- Found the module
           module.gate = true -- let everything flow!
           found_resnet_blocks[#found_resnet_blocks+1] = module
        end
    end)
    return found_resnet_blocks
end
function delete_layer(block)
   if torch.typename(block) == 'nn.Sequential' then
      local concat = block.modules[1]
      assert(torch.typename(concat) == 'nn.ConcatTable')
      -- The branches are the two sides. Drop the first one; the second one will
      -- be the identity layer or the downsampling layer.
      --[[
      if torch.typename(concat.modules[1]) == 'nn.Identity' then
      print "Dropping the entire thing"
      block.modules = {}
      block.gradInput = torch.Tensor()
      block.output = torch.Tensor()
      else
      --]]
      table.remove(concat.modules, 1)
      --concat.gradInput = torch.Tensor()
      --concat.output = torch.Tensor()
      --end
   elseif torch.typename(block) == 'nn.ResidualDrop' then
      block.SUPPRESS_DURING_TESTING = true
   else
      error 'Unknown layer type'
   end
end
function permute_layers(block1, block2)
    local tmp_mods = block2.modules
    block2.modules = block1.modules
    block1.modules = tmp_mods
    block1.output = {}
    block2.output = {}
end
function duplicate_layer(block)
   block.modules[#block.modules+1] = block:clone()
   block.output = {}
end

-- surgery time!
local blocks = gather_residual_blocks(model)
print("There are ", #blocks, " blocks in total")

if opt.duplicateBlock ~= 'none' then
    for _,blk in ipairs(string.split(opt.duplicateBlock, ",")) do
       print("Duplicating block "..blk)
       duplicate_layer(blocks[tonumber(blk)])
    end
end
if opt.permuteBlock ~= 'none' then
    for _,pair in ipairs(string.split(opt.permuteBlock, ",")) do
        local a,b = unpack(string.split(pair, "-"))
        print("Permuting blocks "..a.." and "..b)
        permute_layers(blocks[tonumber(a)], blocks[tonumber(b)])
    end
end
if opt.deleteBlock ~= 'none' then
    for _,blk in ipairs(string.split(opt.deleteBlock, ",")) do
        print("Deleting block ",blk)
        delete_layer(blocks[tonumber(blk)])
    end
end


-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

local top1Err, top5Err, classAccuracy = trainer:test(0, valLoader)

local f = io.open(opt.saveClassAccuracy, "w")
f:write(msgpack.pack(classAccuracy))
f:close()

print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
