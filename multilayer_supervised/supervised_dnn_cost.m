function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%
for i=1:(numHidden+1)
    if i==1 %input unit
        hAct{i}=sigmoid(bsxfun(@plus,stack{i}.W*data,stack{i}.b));
    elseif i==(numHidden+1) %output unit
        hAct{i}=softmax(bsxfun(@plus,stack{i}.W*hAct{i-1},stack{i}.b));
    else
        hAct{i}=sigmoid(stack{i}.W*hAct{i-1}+stack{i}.b);
    end
end
%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
ind=sub2ind(size(hAct{i}),1:length(labels),labels);
cost=-sum(log(hAct{i}(ind)));
%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
while i>0
    if i==(numHidden+1)
        gradStack{i}=hAct{i};
        gradStack{i}(ind)=sum(gradStack{i}(ind)-1)';
    else
        gradStack{i}=stack{i}.W'*gradStack{i+1}.*hAct{i}.*(1-hAct{i});
    end
    i=i-1;
end
%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



