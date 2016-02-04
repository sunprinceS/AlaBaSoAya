--[[

  A Binary Tree-LSTM with input at the leaf nodes.

--]]

local BinaryTreeLSTM, parent = torch.class('treelstm.BinaryTreeLSTM', 'treelstm.TreeLSTM')

function BinaryTreeLSTM:__init(config)
  parent.__init(self, config)
  self.gate_output = config.gate_output
  if self.gate_output == nil then self.gate_output = true end

  -- a function that instantiates an output module that takes the hidden state h as input
  self.criterion = config.criterion

  -- leaf input module
  self.leaf_module = self:new_leaf_module()
  self.leaf_modules = {}

  -- composition module
  self.composer = self:new_composer()
  self.composers = {}


  -- output module
  self.output_module = self:new_output_module()
  self.output_modules = {}

  self.aspect_module = self:new_aspect_module()
  self.aspect_modules = {}

  self.absa_module = self:new_absa_module()
  self.absa_modules = {}
end

function BinaryTreeLSTM:new_leaf_module()
  local input = nn.Identity()()
  local c = nn.Linear(self.in_dim, self.mem_dim)(input)
  local h
  if self.gate_output then
    local o = nn.Sigmoid()(nn.Linear(self.in_dim, self.mem_dim)(input))
    h = nn.CMulTable(){o, nn.Tanh()(c)}
  else
    h = nn.Tanh()(c)
  end

  local leaf_module = nn.gModule({input}, {c, h})
  if self.leaf_module ~= nil then
    share_params(leaf_module, self.leaf_module)
  end
  return leaf_module
end

function BinaryTreeLSTM:new_composer()
  local lc, lh = nn.Identity()(), nn.Identity()()
  local rc, rh = nn.Identity()(), nn.Identity()()
  local new_gate = function()
    return nn.CAddTable(){
      nn.Linear(self.mem_dim, self.mem_dim)(lh),
      nn.Linear(self.mem_dim, self.mem_dim)(rh)
    }
  end

  local i = nn.Sigmoid()(new_gate())    -- input gate
  local lf = nn.Sigmoid()(new_gate())   -- left forget gate
  local rf = nn.Sigmoid()(new_gate())   -- right forget gate
  local update = nn.Tanh()(new_gate())  -- memory cell update vector
  local c = nn.CAddTable(){             -- memory cell
      nn.CMulTable(){i, update},
      nn.CMulTable(){lf, lc},
      nn.CMulTable(){rf, rc}
    }

  local h
  if self.gate_output then
    local o = nn.Sigmoid()(new_gate()) -- output gate
    h = nn.CMulTable(){o, nn.Tanh()(c)}
  else
    h = nn.Tanh()(c)
  end
  local composer = nn.gModule(
    {lc, lh, rc, rh},
    {c, h})

  if self.composer ~= nil then
    share_params(composer, self.composer)
  end
  return composer
end

function BinaryTreeLSTM:new_aspect_module()
    local input = nn.Identity()()
    local aspect_copy = nn.Copy(int, int)(input)
    local aspect_module = nn.gModule({input}, {aspect_copy})
    if self.aspect_module ~= nil then
        share_params(aspect_module, self.aspect_module)
    end
    return aspect_module
end

function BinaryTreeLSTM:new_absa_module()
    local aspect = nn.Identity()()
    local sent = nn.Identity()()
    local join = nn.JoinTable(1)({aspect, sent})
    local dropout1 = nn.Dropout()(join)
    --[
    local hidden1  = nn.Linear((self.aspect_num+self.mem_dim), self.mlp_hidden_units)(dropout1)
    local actfunc1 = nn.ReLU()(hidden1)
    local dropout2 = nn.Dropout()(actfunc1)
    local hidden2 = nn.Linear(self.mlp_hidden_units, self.mlp_hidden_units)(dropout2)
    local actfunc2 = nn.ReLU()(hidden2)
    local dropout3 = nn.Dropout()(actfunc2)
    local hidden3 = nn.Linear(self.mlp_hidden_units, self.mlp_hidden_units)(dropout3)
    local actfunc3 = nn.ReLU()(hidden3)
    local dropout4 = nn.Dropout()(actfunc3)
    local out = nn.Linear(self.mlp_hidden_units, self.num_polarities)(dropout4)
    local out_prob = nn.LogSoftMax()(out)
    local absa_module = nn.gModule({aspect, sent}, {out_prob})
    --]]
    if self.absa_module ~= nil then
        share_params(absa_module, self.absa_module)
    end
    return absa_module
end

function BinaryTreeLSTM:new_output_module()
  local output_module = nn.Sequential()
    output_module:add(nn.Dropout())
    output_module
      :add(nn.Linear(self.mem_dim, self.num_polarities))
      :add(nn.LogSoftMax())
  if self.output_module ~= nil then
    share_params(output_module, self.output_module)
  end
  return output_module
end

function BinaryTreeLSTM:forward(tree, inputs, aspects, root)
  local lloss, rloss = 0, 0
  if tree.num_children == 0 then
    self:allocate_module(tree, 'leaf_module')
    tree.state = tree.leaf_module:forward(inputs[tree.leaf_idx])
  else
    self:allocate_module(tree, 'composer')

    -- get child hidden states
    local lvecs, lloss = self:forward(tree.children[1], inputs, nil, false)
    local rvecs, rloss = self:forward(tree.children[2], inputs, nil, false)
    local lc, lh = self:unpack_state(lvecs)
    local rc, rh = self:unpack_state(rvecs)

    -- compute state and output
    tree.state = tree.composer:forward{lc, lh, rc, rh}
  end

  local loss
  if self.output_module ~= nil and not root then
    self:allocate_module(tree, 'output_module')
    tree.output = tree.output_module:forward(tree.state[2])
    if self.train then
      loss = self.criterion:forward(tree.output, tree.gold_label) + lloss + rloss
    end
  end

  if root then
      self:allocate_module(tree, 'aspect_module')
      tree.aspect_tensor = tree.aspect_module:forward(aspects)
      self:allocate_module(tree, 'absa_module')
      --print(tree.aspect_tensor)
      --print(tree.state[2])
      tree.final_output = tree.absa_module:forward({tree.aspect_tensor, (tree.state[2])})
      if self.train then
          loss = self.criterion:forward(tree.final_output, tree.gold_label) + lloss + rloss
      end
  end

  return tree.state, loss
end

function BinaryTreeLSTM:backward(tree, inputs, aspects, grad)
  local grad_inputs = torch.Tensor(inputs:size())
  self:_backward(tree, inputs, aspects, grad, grad_inputs, true)
  return grad_inputs
end

function BinaryTreeLSTM:_backward(tree, inputs, aspects, grad, grad_inputs, root)
  local output_grad = self.mem_zeros
  local absa_grad
  if root then
      absa_grad = tree.absa_module:backward(
        {tree.aspect_tensor, (tree.state[2])},
        self.criterion:backward(tree.final_output, tree.gold_label))
      --print(absa_grad)
      --print(grad)
      aspect_grad = tree.aspect_module:backward(
        aspects, absa_grad[1])
      self:free_module(tree, 'absa_module')
      self:free_module(tree, 'aspect_module')
  end

  if not root and tree.output ~= nil and tree.gold_label ~= nil then
    output_grad = tree.output_module:backward(
      tree.state[2], self.criterion:backward(tree.output, tree.gold_label))
    self:free_module(tree, 'output_module')
  end

  if tree.num_children ~= 0 and root then
    local lc, lh, rc, rh = self:get_child_states(tree)
    local composer_grad = tree.composer:backward(
      {lc, lh, rc, rh},
      {grad[1], grad[2] + absa_grad[2]})
    self:free_module(tree, 'composer')
    -- backward propagate to children
    self:_backward(tree.children[1], inputs, nil, {composer_grad[1], composer_grad[2]}, grad_inputs, false)
    self:_backward(tree.children[2], inputs, nil, {composer_grad[3], composer_grad[4]}, grad_inputs, false)

  elseif tree.num_children ~= 0 then
    local lc, lh, rc, rh = self:get_child_states(tree)
    local composer_grad = tree.composer:backward(
      {lc, lh, rc, rh},
      {grad[1], grad[2] + output_grad})
    self:free_module(tree, 'composer')
    -- backward propagate to children
    self:_backward(tree.children[1], inputs, nil, {composer_grad[1], composer_grad[2]}, grad_inputs, false)
    self:_backward(tree.children[2], inputs, nil, {composer_grad[3], composer_grad[4]}, grad_inputs, false)

  else
    grad_inputs[tree.leaf_idx] = tree.leaf_module:backward(
      inputs[tree.leaf_idx],
      {grad[1], grad[2] + output_grad})
    self:free_module(tree, 'leaf_module')

  end

  tree.state = nil
  tree.output = nil
  tree.final_output = nil
  tree.aspect_tensor = nil
end

function BinaryTreeLSTM:parameters()
  local params, grad_params = {}, {}
  local cp, cg = self.composer:parameters()
  tablex.insertvalues(params, cp)
  tablex.insertvalues(grad_params, cg)
  local lp, lg = self.leaf_module:parameters()
  tablex.insertvalues(params, lp)
  tablex.insertvalues(grad_params, lg)
  if self.output_module ~= nil then
    local op, og = self.output_module:parameters()
    tablex.insertvalues(params, op)
    tablex.insertvalues(grad_params, og)
  end
  if self.aspect_module ~= nil then
    local ap, ag = self.aspect_module:parameters()
    tablex.insertvalues(params, ap)
    tablex.insertvalues(grad_params, ag)
  end
  if self.absa_module ~= nil then
    local absap, absag = self.absa_module:parameters()
    tablex.insertvalues(params, absap)
    tablex.insertvalues(grad_params, absag)
  end
  return params, grad_params
end

--
-- helper functions
--

function BinaryTreeLSTM:unpack_state(state)
  local c, h
  if state == nil then
    c, h = self.mem_zeros, self.mem_zeros
  else
    c, h = unpack(state)
  end
  return c, h
end

function BinaryTreeLSTM:get_child_states(tree)
  local lc, lh, rc, rh
  if tree.children[1] ~= nil then
    lc, lh = self:unpack_state(tree.children[1].state)
  end

  if tree.children[2] ~= nil then
    rc, rh = self:unpack_state(tree.children[2].state)
  end
  return lc, lh, rc, rh
end

function BinaryTreeLSTM:clean(tree)
  tree.state = nil
  tree.output = nil
  tree.final_output = nil
  self:free_module(tree, 'leaf_module')
  self:free_module(tree, 'composer')
  self:free_module(tree, 'output_module')
  self.free_module(tree, 'aspect_module')
  self.free_module(tree, 'absa_module')
  for i = 1, tree.num_children do
    self:clean(tree.children[i])
  end
end
