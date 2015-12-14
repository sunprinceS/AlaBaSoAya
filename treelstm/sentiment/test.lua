--[[

  Tree-LSTM training script for sentiment classication on the Stanford
  Sentiment Treebank

--]]

require('..')

function accuracy(pred, gold)
  return torch.eq(pred, gold):sum() / pred:size(1)
end

-- read command line arguments
local args = lapp [[
Testing script for sentiment classification on the SST dataset.
  -t,--dataset (default restaurant)  Datasets: [restaurant, laptop]
  <model> (string) Model path
]]
print(args.model)

local model_class = treelstm.TreeLSTMSentiment

-- binary or fine-grained subtask
local fine_grained = not args.binary

-- directory containing dataset files
local data_dir
if args.dataset == 'restaurant' then
  data_dir = 'data/absa_restaurant/'
elseif args.dataset == 'laptop' then
  data_dir = 'data/absa_laptop/'
end

-- load vocab
local vocab = treelstm.Vocab(data_dir .. 'vocab-cased.txt')

-- load embeddings
print('loading word embeddings')
local emb_dir = 'data/glove/'
local emb_prefix = emb_dir .. 'glove.840B'
local emb_vocab, emb_vecs = treelstm.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.300d.th')
local emb_dim = emb_vecs:size(2)

-- use only vectors in vocabulary (not necessary, but gives faster training)
local num_unk = 0
local vecs = torch.Tensor(vocab.size, emb_dim)
for i = 1, vocab.size do
  local w = string.gsub(vocab:token(i), '\\', '') -- remove escape characters
  if emb_vocab:contains(w) then
    vecs[i] = emb_vecs[emb_vocab:index(w)]
  else
    num_unk = num_unk + 1
    vecs[i]:uniform(-0.05, 0.05)
  end
end
print('unk count = ' .. num_unk)
emb_vocab = nil
emb_vecs = nil
collectgarbage()

-- load datasets
print('loading datasets')
local dev_dir = data_dir .. 'dev/'
--local test_dir = data_dir .. 'test/'
local dependency = true
local dev_dataset = treelstm.read_sentiment_dataset(dev_dir, vocab, fine_grained, dependency)
--local test_dataset = treelstm.read_sentiment_dataset(test_dir, vocab, fine_grained, dependency)

printf('num dev   = %d\n', dev_dataset.size)
--printf('num test  = %d\n', test_dataset.size)

-- load model
local model = model_class.load(args.model)

-- print information
header('model configuration')
model:print_config()

local dev_features = model:get_features(dev_dataset)
print(dev_features)
--local dev_predictions = model:predict_dataset(dev_dataset)
--local dev_score = accuracy(dev_predictions, dev_dataset.labels)
--printf('-- dev score: %.4f\n', dev_score)


-- evaluate
--[[
header('Evaluating on test set')
printf('-- using model with dev score = %.4f\n', best_dev_score)
local test_predictions = best_dev_model:predict_dataset(test_dataset)
printf('-- test score: %.4f\n', accuracy(test_predictions, test_dataset.labels))
--]]

-- create predictions and models directories if necessary
--[[
if lfs.attributes(treelstm.predictions_dir) == nil then
  lfs.mkdir(treelstm.predictions_dir)
end
--]]

-- get paths
--[[
local file_idx = 1
local subtask = fine_grained and '5class' or '2class'
local predictions_save_path
while true do
  predictions_save_path = string.format(
    treelstm.predictions_dir .. '/sent-%s.%s.%dl.%dd.%d.pred', args.model, subtask, args.layers, args.dim, file_idx)
  --if lfs.attributes(predictions_save_path) == nil and lfs.attributes(model_save_path) == nil then
  if lfs.attributes(predictions_save_path) == nil then
    break
  end
  file_idx = file_idx + 1
end
--]]

-- write predictions to disk
--[[
local predictions_file = torch.DiskFile(predictions_save_path, 'w')
print('writing predictions to ' .. predictions_save_path)
for i = 1, test_predictions:size(1) do
  predictions_file:writeInt(test_predictions[i])
end
predictions_file:close()
--]]

