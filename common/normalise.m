function [p, lnp]=normalise(logp,dim)
%returns exp(logp)/sum(exp(logp)) without numerical problems for a NxD
%matrix
max_logp=max(logp,[],dim);
logp=bsxfun(@minus,logp,max_logp);
p=exp(logp);
C = sum(p,dim);
lnC = log(C);
p=bsxfun(@rdivide,p,C);
lnp = bsxfun(@minus,logp,lnC);