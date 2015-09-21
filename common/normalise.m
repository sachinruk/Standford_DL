function p=normalise(logp,dim)
%returns exp(logp)/sum(exp(logp)) without numerical problems for a NxD
%matrix
max_logp=max(logp,[],dim);
logp=bsxfun(@minus,logp,max_logp);
p=exp(logp);
p=bsxfun(@rdivide,p,sum(p,dim));