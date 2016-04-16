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
    local found_resnet_blocks = {}
    model:apply(function(module)
        if (torch.typename(module)=='nn.Sequential' 
            and module.modules 
            and torch.typename(module.modules[#module.modules-1])=='nn.CAddTable') then
            found_resnet_blocks[#found_resnet_blocks+1] = module
        end
    end)
    return found_resnet_blocks
end

if opt.deleteBlock ~= 0 then
    print("Deleting block ",opt.deleteBlock)
    local block = gather_residual_blocks(model)[opt.deleteBlock]
    block.modules = {}
    block.gradInput = torch.Tensor()
    block.output = torch.Tensor()
end


-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)


local top1Err, top5Err, classAccuracy = trainer:test(0, valLoader)

local f = io.open(opt.saveClassAccuracy, "w")
f:write(msgpack.pack(classAccuracy))
f:close()

print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
