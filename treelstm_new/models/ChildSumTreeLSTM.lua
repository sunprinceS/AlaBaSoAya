--[[

  A Child-Sum Tree-LSTM with input at each node.

--]]

local ChildSumTreeLSTM, parent = torch.class('treelstm.ChildSumTreeLSTM', 'treelstm.TreeLSTM')

function ChildSumTreeLSTM:__init(config)
  parent.__init(self, config)
  self.gate_output = config.gate_output
  if self.gate_output == nil then self.gate_output = true end

  -- a function that instantiates an output module that takes the hidden state h as input
  --self.output_module_fn = config.output_module_fn
  self.criterion = config.criterion

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

function ChildSumTreeLSTM:new_composer()
  local input = nn.Identity()()
  local child_c = nn.Identity()()
  local child_h = nn.Identity()()
  local child_h_sum = nn.Sum(1)(child_h)

  local i = nn.Sigmoid()(
    nn.CAddTable(){
      nn.Linear(self.in_dim, self.mem_dim)(input),
      nn.Linear(self.mem_dim, self.mem_dim)(child_h_sum)
    })
  local f = nn.Sigmoid()(
    treelstm.CRowAddTable(){
      nn.TemporalConvolution(self.mem_dim, self.mem_dim, 1)(child_h),
      nn.Linear(self.in_dim, self.mem_dim)(input),
    })
  local update = nn.Tanh()(
    nn.CAddTable(){
      nn.Linear(self.in_dim, self.mem_dim)(input),
      nn.Linear(self.mem_dim, self.mem_dim)(child_h_sum)
    })
  local c = nn.CAddTable(){
      nn.CMulTable(){i, update},
      nn.Sum(1)(nn.CMulTable(){f, child_c})
    }

  local h
  if self.gate_output then
    local o = nn.Sigmoid()(
      nn.CAddTable(){
        nn.Linear(self.in_dim, self.mem_dim)(input),
        nn.Linear(self.mem_dim, self.mem_dim)(child_h_sum)
      })
    h = nn.CMulTable(){o, nn.Tanh()(c)}
  else
    h = nn.Tanh()(c)
  end

  local composer = nn.gModule({input, child_c, child_h}, {c, h})
  if self.composer ~= nil then
    share_params(composer, self.composer)
  end
  return composer
end

function ChildSumTreeLSTM:new_output_module()
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

function ChildSumTreeLSTM:new_aspect_module()
    local input = nn.Identity()()
    local aspect_copy = nn.Copy(int, int)(input)
    local aspect_module = nn.gModule({input}, {aspect_copy})
    if self.aspect_module ~= nil then
        share_params(aspect_module, self.aspect_module)
    end
    return aspect_module
end

function ChildSumTreeLSTM:new_absa_module()
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

function ChildSumTreeLSTM:forward(tree, inputs, aspects, root)
  local loss = 0
  for i = 1, tree.num_children do
    local _, child_loss = self:forward(tree.children[i], inputs, nil, false)
    loss = loss + child_loss
  end
  local child_c, child_h = self:get_child_states(tree)
  self:allocate_module(tree, 'composer')
  tree.state = tree.composer:forward{inputs[tree.idx], child_c, child_h}

  if self.output_module ~= nil and not root then
    self:allocate_module(tree, 'output_module')
    tree.output = tree.output_module:forward(tree.state[2])
    if self.train and tree.gold_label ~= nil then
      loss = loss + self.criterion:forward(tree.output, tree.gold_label)
    end
  elseif self.output_module ~= nil then
    self:allocate_module(tree, 'aspect_module')
    tree.aspect_tensor = tree.aspect_module:forward(aspects)
    self:allocate_module(tree, 'absa_module')
    tree.final_output = tree.absa_module:forward({tree.aspect_tensor, (tree.state[2])})
    if self.train and tree.gold_label ~= nil then
      loss = loss + self.criterion:forward(tree.final_output, tree.gold_label)
    end
  end
  return tree.state, loss
end

function ChildSumTreeLSTM:backward(tree, inputs, aspects, grad)
  local grad_inputs = torch.Tensor(inputs:size())
  self:_backward(tree, inputs, aspects, grad, grad_inputs, true)
  return grad_inputs
end

function ChildSumTreeLSTM:_backward(tree, inputs, aspects, grad, grad_inputs, root)
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
  tree.final_output = nil
  tree.aspect_tensor = nil

  if not root and tree.output ~= nil and tree.gold_label ~= nil then
    output_grad = tree.output_module:backward(
      tree.state[2], self.criterion:backward(tree.output, tree.gold_label))
    self:free_module(tree, 'output_module')
  end
  tree.output = nil

  local composer_grad
  if root then
      local child_c, child_h = self:get_child_states(tree)
      composer_grad = tree.composer:backward(
        {inputs[tree.idx], child_c, child_h},
        {grad[1], grad[2] + absa_grad[2]})
  else
      local child_c, child_h = self:get_child_states(tree)
      composer_grad = tree.composer:backward(
        {inputs[tree.idx], child_c, child_h},
        {grad[1], grad[2] + output_grad})
  end
  self:free_module(tree, 'composer')
  tree.state = nil

  grad_inputs[tree.idx] = composer_grad[1]
  local child_c_grads, child_h_grads = composer_grad[2], composer_grad[3]
  for i = 1, tree.num_children do
    self:_backward(tree.children[i], inputs, nil, {child_c_grads[i], child_h_grads[i]}, grad_inputs, false)
  end
end

function ChildSumTreeLSTM:clean(tree)
  self:free_module(tree, 'composer')
  self:free_module(tree, 'output_module')
  self:free_module(tree, 'absa_module')
  self:free_module(tree, 'aspect_module')
  tree.state = nil
  tree.output = nil
  tree.final_output = nil
  tree.aspect_tensor = nil
  for i = 1, tree.num_children do
    self:clean(tree.children[i])
  end
end

function ChildSumTreeLSTM:parameters()
  local params, grad_params = {}, {}
  local cp, cg = self.composer:parameters()
  tablex.insertvalues(params, cp)
  tablex.insertvalues(grad_params, cg)
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

function ChildSumTreeLSTM:get_child_states(tree)
  local child_c, child_h
  if tree.num_children == 0 then
    child_c = torch.zeros(1, self.mem_dim)
    child_h = torch.zeros(1, self.mem_dim)
  else
    child_c = torch.Tensor(tree.num_children, self.mem_dim)
    child_h = torch.Tensor(tree.num_children, self.mem_dim)
    for i = 1, tree.num_children do
       child_c[i], child_h[i] = unpack(tree.children[i].state)
    end
  end
  return child_c, child_h
end
