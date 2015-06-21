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
hAct = cell(numHidden+2, 1);
gradStack = cell(numHidden+1, 1);
% gradStackDelta = cell(numHidden+2, 1);
hAct{1}=data;
%% forward prop
%%% YOUR CODE HERE %%%
for i=2:(numHidden+2)
    z=bsxfun(@plus,stack{i-1}.W*hAct{i-1},stack{i-1}.b);
%     if i==1 %input unit
%         hAct{i}=sigmoid(bsxfun(@plus,stack{i}.W*data,stack{i}.b));
    if i==(numHidden+2) %output unit
        hAct{i}=softmax(z);
    else
        hAct{i}=sigmoid(z);
    end
end
%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  pred_prob=hAct{i};
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
ind=sub2ind(size(hAct{i}),labels',1:length(labels));
cost=-sum(log(hAct{i}(ind)));
%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
gradDelta=hAct{i};
gradDelta(ind)=gradDelta(ind)-1;
while i>1
    i=i-1;
%     if i==(numHidden+1)
        
%         gradStack{i}=sum(gradStack{i},2);
%     else
    gradStack{i}.b=sum(gradDelta,2);
    gradStack{i}.W=gradDelta*hAct{i}';
    gradDelta=stack{i}.W'*gradDelta.*hAct{i}.*(1-hAct{i});
%     end
    
    
end
%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
% i=1;
% while i<=(numHidden+1)
%     gradStack{i}.b=sum(gradStackDelta{i+1},2);
%     gradStack{i}.W=gradStackDelta{i+1}*hAct{i}';
%     i=i+1;
% end

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



